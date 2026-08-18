from dataclasses import dataclass

from sympy import Abs, Add, Expr, Max, Min, exp

from ..reaction import Reaction
from ..species.registry import PROPERTY_REGISTRY
from ..state import StateVariables


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
PRESSURE_PA_PER_BAR = 1.0e5
MIN_STEAM_PARTIAL_PRESSURE_BAR = 1.0e-2
STEAM_REFORMING_H2O_ORDER = 1.596
NI_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").mw

NUMAGUCHI_RATE_COEFFICIENTS = {
    "smr": 3.65e5,
    "wgs": 2.45e5,
}
NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 42800.0,
    "wgs": 54531.0,
}
SMR_EQUILIBRIUM_INTERCEPT = 30.114
SMR_EQUILIBRIUM_TEMPERATURE_TERM = -26830.0
WGS_EQUILIBRIUM_INTERCEPT = -4.036
WGS_EQUILIBRIUM_TEMPERATURE_TERM = 4400.0


@dataclass(frozen=True)
class NumaguchiTerms:
    temperature_k: Expr
    p_ch4_bar: Expr
    p_co_bar: Expr
    p_co2_bar: Expr
    p_h2_bar: Expr
    p_h2o_bar: Expr
    p_h2o_bar_safe: Expr
    catalyst_mass_density_kg_per_m3: Expr

def _total_gas_concentration(states: StateVariables) -> Expr:
    gas_concentrations = tuple(
        concentration
        for species_id, concentration in states.concentrations.items()
        if PROPERTY_REGISTRY.get_record(species_id).phase == "gas"
    )

    if not gas_concentrations:
        raise ValueError("Numaguchi kinetics require at least one gas species.")

    return Add(*gas_concentrations)

def xu_froment_terms(states: StateVariables) -> NumaguchiTerms:
    temperature_k = states.temperature
    pressure_pa = states.pressure
    total_gas_concentration = _total_gas_concentration(states)

    def partial_pressure(species_id: str) -> Expr:
        return (
            pressure_pa
            * states.concentration(species_id)
            / total_gas_concentration
        )

    p_ch4_pa = partial_pressure("CH4")
    p_co_pa = partial_pressure("CO")
    p_co2_pa = partial_pressure("CO2")
    p_h2_pa = partial_pressure("H2")
    p_h2o_pa = partial_pressure("H2O")


def smr_fw_rate(states: StateVariables) -> Expr:
    terms = xu_froment_terms(states)
    driving_force = terms.p_ch4_pa * terms.p_h2o_pa
    return (
        _common_rate_factor(terms, "smr", Rational(5, 2), 10.0**-2.5)
        * driving_force
    )


def build_smr_fw(states: StateVariables) -> Reaction:
    return Reaction(
        name="smr_fw",
        family="xu_froment",
        reactants={"CH4": 1, "H2O": 1},
        products={"CO": 1, "H2": 3},
        catalysts=("Ni",),
        rate=smr_fw_rate(states),
    )


def smr_bw_rate(states: StateVariables) -> Expr:
    terms = xu_froment_terms(states)
    driving_force = (
        Pow(terms.p_h2_pa, 3)
        * terms.p_co_pa
        / (1e10 * _eq_const_smr(terms.temperature_k))
    )
    return (
        _common_rate_factor(terms, "smr", Rational(5, 2), 10.0**-2.5)
        * driving_force
    )


def build_smr_bw(states: StateVariables) -> Reaction:
    return Reaction(
        name="smr_bw",
        family="xu_froment",
        reactants={"CO": 1, "H2": 3},
        products={"CH4": 1, "H2O": 1},
        catalysts=("Ni",),
        rate=smr_bw_rate(states),
    )






REACTIONS = {
    #"wgs_fw": build_wgs_fw,
    "smr_fw": build_smr_fw,
    #"wgs_bw": build_wgs_bw,
    "smr_bw": build_smr_bw,
}