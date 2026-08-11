from dataclasses import dataclass

from sympy import Abs, Add, Expr, Max, Min, exp, sympify

from ..reaction import Reaction
from ..species.registry import PROPERTY_REGISTRY
from ..state import StateVariables


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
DENOMINATOR_FLOOR = 1.0e-16
POS_EPS = 1.0e-10
F_GATE = 1.0e-3
Y_GATE = 1.0e-4

ONE_THIRD = 1.0 / 3.0
TWO_THIRDS = 2.0 / 3.0

CS_MOL_PER_M3 = {
    "H2": 89960.0,
    "CO": 89960.0,
    "O2": 151200.0,
}
R0_M = {
    "H2": 3.13e-8,
    "CO": 3.13e-8,
    "O2": 5.8e-7,
}
K0_M_PER_S = {
    "H2": 9.0e-4,
    "CO": 3.5e-3,
    "O2": 1.2e-3,
}
ACTIVATION_ENERGY_J_PER_MOL = {
    "H2": 30000.0,
    "CO": 45000.0,
    "O2": 7000.0,
}
REACTION_ORDER = {
    "H2": 0.60,
    "CO": 0.65,
    "O2": 0.90,
}
D0_M2_PER_S = {
    "H2": 1.7e-3,
    "CO": 7.4e6,
    "O2": 1.0,
}
ED_J_PER_MOL = {
    "H2": 150000.0,
    "CO": 300000.0,
    "O2": 0.0,
}
KX = {
    "H2": 5.0,
    "CO": 15.0,
    "O2": 0.0,
}
KXE_J_PER_MOL = {
    "H2": 0.0,
    "CO": 0.0,
    "O2": 0.0,
}
B = {
    "H2": 1.0,
    "CO": 1.0,
    "O2": 2.0,
}

# Two-term rational approximations a*x/(1+b*|x|) + c*x/(1+d*|x|),
# fitted over [0, 1].
RATIONAL_POWER_COEFFICIENTS = {
    ONE_THIRD: (1.36709714, 1.19782338, 38.81369122, 103.2164461),
    TWO_THIRDS: (1.10119253, 0.31786513, 3.171007230, 18.51100221),
    0.60: (1.13885038, 0.42300595, 4.952297600, 24.11159535),
    0.65: (1.10997490, 0.34224903, 3.545189580, 19.72954101),
    0.90: (1.01682681, 0.06978578, 0.468259710, 8.536784090),
}


@dataclass(frozen=True)
class MedranoTerms:
    temperature_k: Expr
    total_gas_conc_molm3: Expr
    gas_mole_fraction: Expr
    ni_conc_molm3: Expr
    nio_conc_molm3: Expr
    total_solid_inventory_molm3: Expr
    frac_reduced: Expr
    frac_oxidised: Expr


@dataclass(frozen=True)
class MedranoReactionState:
    conversion: Expr
    unreacted_fraction: Expr
    total_solid_inventory_molm3: Expr


def _total_gas_concentration(states: StateVariables) -> Expr:
    gas_concentrations = tuple(
        concentration
        for species_id, concentration in states.concentrations.items()
        if PROPERTY_REGISTRY.get_record(species_id).phase == "gas"
    )

    if not gas_concentrations:
        raise ValueError("Medrano kinetics require at least one gas species.")

    return Add(*gas_concentrations)


def _available_expr(x: Expr) -> Expr:
    return Max(x, 0)


def _bounded_fraction_expr(x: Expr) -> Expr:
    return Min(1, Max(x, 0))


def _availability_gate_expr(x: Expr, gate: float) -> Expr:
    available = _available_expr(x)
    return available / (available + gate)


def medrano_terms(
    states: StateVariables,
    gas_species_id: str,
) -> MedranoTerms:
    temperature_k = states.temperature
    c_ni = _available_expr(states.concentration("Ni"))
    c_nio = _available_expr(states.concentration("NiO"))
    c_solid_total = c_ni + c_nio
    c_solid_denominator = c_solid_total + POS_EPS

    return MedranoTerms(
        temperature_k=temperature_k,
        total_gas_conc_molm3=(
            states.pressure / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        ),
        gas_mole_fraction=(
            states.concentration(gas_species_id) / _total_gas_concentration(states)
        ),
        ni_conc_molm3=c_ni,
        nio_conc_molm3=c_nio,
        total_solid_inventory_molm3=c_solid_total,
        frac_reduced=c_ni / c_solid_denominator,
        frac_oxidised=c_nio / c_solid_denominator,
    )


def _medrano_reaction_state_expr(
    comp_key: str,
    terms: MedranoTerms,
) -> MedranoReactionState:
    if comp_key == "O2":
        return MedranoReactionState(
            conversion=terms.frac_oxidised,
            unreacted_fraction=terms.frac_reduced,
            total_solid_inventory_molm3=terms.total_solid_inventory_molm3,
        )
    if comp_key in {"H2", "CO"}:
        return MedranoReactionState(
            conversion=terms.frac_reduced,
            unreacted_fraction=terms.frac_oxidised,
            total_solid_inventory_molm3=terms.total_solid_inventory_molm3,
        )
    raise KeyError(f"Unsupported Medrano AN component key: {comp_key}")


def _rational_power_expr(power: float, x: Expr) -> Expr:
    a, b, c, d = RATIONAL_POWER_COEFFICIENTS[power]
    raw_value = (
        a * x / (1.0 + b * Abs(x))
        + c * x / (1.0 + d * Abs(x))
    )
    raw_value_at_one = a / (1.0 + b) + c / (1.0 + d)
    return raw_value / raw_value_at_one


def _gas_concentration_power_expr(
    *,
    total_gas_concentration_molm3: Expr,
    gas_mole_fraction: Expr,
    power: float,
) -> Expr:
    return total_gas_concentration_molm3**power * _rational_power_expr(
        power,
        _available_expr(gas_mole_fraction),
    )


def k_expr(comp_key: str, *, temperature_k: Expr) -> Expr:
    return K0_M_PER_S[comp_key] * exp(
        -ACTIVATION_ENERGY_J_PER_MOL[comp_key]
        / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def D_expr(
    comp_key: str,
    *,
    temperature_k: Expr,
    conversion: Expr,
) -> Expr:
    diffusivity = sympify(D0_M2_PER_S[comp_key])
    activation_energy = ED_J_PER_MOL[comp_key]
    if activation_energy != 0.0:
        diffusivity *= exp(
            -activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
        )

    kx = KX[comp_key]
    if kx != 0.0:
        conversion_factor = kx
        kxe = KXE_J_PER_MOL[comp_key]
        if kxe != 0.0:
            conversion_factor *= exp(
                -kxe / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
            )
        diffusivity *= exp(-conversion_factor * conversion)

    return diffusivity


def _denominator_safe_expr(denominator: Expr) -> Expr:
    return Max(denominator, DENOMINATOR_FLOOR)


def _medrano_conversion_rate_expr(
    comp_key: str,
    *,
    temperature_k: Expr,
    total_gas_concentration_molm3: Expr,
    gas_mole_fraction: Expr,
    conversion: Expr,
    unreacted_fraction: Expr,
) -> Expr:
    order = REACTION_ORDER[comp_key]
    k_reaction = k_expr(comp_key, temperature_k=temperature_k)
    conversion_bounded = _bounded_fraction_expr(conversion)
    unreacted_available = _bounded_fraction_expr(unreacted_fraction)
    gas_fraction_available = _available_expr(gas_mole_fraction)
    diffusivity = D_expr(
        comp_key,
        temperature_k=temperature_k,
        conversion=conversion_bounded,
    )
    c_power_kinetic = _gas_concentration_power_expr(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    c_power_diffusive = _gas_concentration_power_expr(
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        power=order,
    )
    f_one_third = _rational_power_expr(ONE_THIRD, unreacted_available)
    f_two_thirds = _rational_power_expr(TWO_THIRDS, unreacted_available)
    numerator = (
        3.0
        * B[comp_key]
        * f_two_thirds
        * k_reaction
        * c_power_kinetic
        * diffusivity
        * c_power_diffusive
        / (R0_M[comp_key] * CS_MOL_PER_M3[comp_key])
    )
    denominator = (
        diffusivity * c_power_diffusive
        + R0_M[comp_key]
        * k_reaction
        * c_power_kinetic
        * (f_one_third - f_two_thirds)
    )
    gas_gate = _availability_gate_expr(gas_fraction_available, Y_GATE)
    solid_gate = _availability_gate_expr(unreacted_available, F_GATE)
    return (
        gas_gate
        * solid_gate
        * numerator
        / _denominator_safe_expr(denominator)
    )


def _medrano_reaction_rate_expr(
    comp_key: str,
    *,
    temperature_k: Expr,
    total_gas_concentration_molm3: Expr,
    gas_mole_fraction: Expr,
    conversion: Expr,
    unreacted_fraction: Expr,
    total_solid_inventory_molm3: Expr,
) -> Expr:
    return total_solid_inventory_molm3 * _medrano_conversion_rate_expr(
        comp_key,
        temperature_k=temperature_k,
        total_gas_concentration_molm3=total_gas_concentration_molm3,
        gas_mole_fraction=gas_mole_fraction,
        conversion=conversion,
        unreacted_fraction=unreacted_fraction,
    )


def _reaction_rate(states: StateVariables, comp_key: str) -> Expr:
    terms = medrano_terms(states, comp_key)
    state = _medrano_reaction_state_expr(comp_key, terms)
    return _medrano_reaction_rate_expr(
        comp_key,
        temperature_k=terms.temperature_k,
        total_gas_concentration_molm3=terms.total_gas_conc_molm3,
        gas_mole_fraction=terms.gas_mole_fraction,
        conversion=state.conversion,
        unreacted_fraction=state.unreacted_fraction,
        total_solid_inventory_molm3=state.total_solid_inventory_molm3,
    )


def reduction_h2_rate(states: StateVariables) -> Expr:
    return _reaction_rate(states, "H2")


def build_reduction_h2(states: StateVariables) -> Reaction:
    return Reaction(
        name="reduction_h2",
        family="medrano",
        reactants={"H2": 1, "NiO": 1},
        products={"Ni": 1, "H2O": 1},
        rate=reduction_h2_rate(states),
    )


def reduction_co_rate(states: StateVariables) -> Expr:
    return _reaction_rate(states, "CO")


def build_reduction_co(states: StateVariables) -> Reaction:
    return Reaction(
        name="reduction_co",
        family="medrano",
        reactants={"CO": 1, "NiO": 1},
        products={"Ni": 1, "CO2": 1},
        rate=reduction_co_rate(states),
    )


def oxidation_o2_rate(states: StateVariables) -> Expr:
    return _reaction_rate(states, "O2")


def build_oxidation_o2(states: StateVariables) -> Reaction:
    return Reaction(
        name="oxidation_o2",
        family="medrano",
        reactants={"O2": 0.5, "Ni": 1},
        products={"NiO": 1},
        rate=oxidation_o2_rate(states),
    )


REACTIONS = {
    "reduction_h2": build_reduction_h2,
    "reduction_co": build_reduction_co,
    "oxidation_o2": build_oxidation_o2,
}
