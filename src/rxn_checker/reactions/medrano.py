"""Medrano nickel redox reaction family."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..model import CaseSymbols, Reaction
from ..species import PROPERTY_REGISTRY

GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
DENOMINATOR_FLOOR, POS_EPS, F_GATE, Y_GATE = 1.0e-16, 1.0e-10, 1.0e-3, 1.0e-4
ONE_THIRD, TWO_THIRDS = 1.0 / 3.0, 2.0 / 3.0
RATIONAL_POWER_COEFFICIENTS = {
    ONE_THIRD: (1.36709714, 1.19782338, 38.81369122, 103.2164461),
    TWO_THIRDS: (1.10119253, 0.31786513, 3.171007230, 18.51100221),
    0.60: (1.13885038, 0.42300595, 4.952297600, 24.11159535),
    0.65: (1.10997490, 0.34224903, 3.545189580, 19.72954101),
    0.90: (1.01682681, 0.06978578, 0.468259710, 8.536784090),
}


@dataclass(frozen=True)
class ComponentParameters:
    cs: float
    r0: float
    k0: float
    activation: float
    order: float
    d0: float
    diffusion_energy: float
    kx: float
    kxe: float
    b: float


PARAMETERS = {
    "H2": ComponentParameters(
        89960.0, 3.13e-8, 9.0e-4, 30000.0, 0.60, 1.7e-3, 150000.0, 5.0, 0.0, 1.0
    ),
    "CO": ComponentParameters(
        89960.0, 3.13e-8, 3.5e-3, 45000.0, 0.65, 7.4e6, 300000.0, 15.0, 0.0, 1.0
    ),
    "O2": ComponentParameters(151200.0, 5.8e-7, 1.2e-3, 7000.0, 0.90, 1.0, 0.0, 0.0, 0.0, 2.0),
}


@dataclass(frozen=True)
class MedranoFamilyTerms:
    temperature_k: sp.Expr
    total_gas_conc_molm3: sp.Expr
    summed_gas_conc_molm3: sp.Expr
    ni_conc_molm3: sp.Expr
    nio_conc_molm3: sp.Expr
    total_solid_inventory_molm3: sp.Expr
    frac_reduced: sp.Expr
    frac_oxidised: sp.Expr


def _available(value):
    return sp.Max(value, 0)


def _bounded(value):
    return sp.Min(1, _available(value))


def _gate(value, width):
    value = _available(value)
    return value / (value + width)


def _total_gas(symbols):
    gases = tuple(
        value
        for species_id, value in symbols.concentrations.items()
        if PROPERTY_REGISTRY.get_record(species_id).phase == "gas"
    )
    if not gases:
        raise ValueError("Medrano kinetics require at least one gas species.")
    return sp.Add(*gases)


def _family_terms(symbols):
    ni, nio = _available(symbols.concentration("Ni")), _available(symbols.concentration("NiO"))
    total = ni + nio
    return MedranoFamilyTerms(
        symbols.temperature,
        symbols.pressure / (GAS_CONSTANT_J_PER_MOL_K * symbols.temperature),
        _total_gas(symbols),
        ni,
        nio,
        total,
        ni / (total + POS_EPS),
        nio / (total + POS_EPS),
    )


def _rational_power(power, value):
    a, b, c, d = RATIONAL_POWER_COEFFICIENTS[power]
    raw = a * value / (1 + b * sp.Abs(value)) + c * value / (1 + d * sp.Abs(value))
    return raw / (a / (1 + b) + c / (1 + d))


def k_expr(comp_key, *, temperature_k):
    params = PARAMETERS[comp_key]
    return params.k0 * sp.exp(-params.activation / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))


def D_expr(comp_key, *, temperature_k, conversion):
    params = PARAMETERS[comp_key]
    value = sp.sympify(params.d0)
    if params.diffusion_energy:
        value *= sp.exp(-params.diffusion_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))
    if params.kx:
        factor = params.kx * (
            sp.exp(-params.kxe / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)) if params.kxe else 1
        )
        value *= sp.exp(-factor * conversion)
    return value


def _reaction_rate(symbols, comp_key, *, shared=None):
    terms = shared or _family_terms(symbols)
    params = PARAMETERS[comp_key]
    conversion, unreacted = (
        (terms.frac_oxidised, terms.frac_reduced)
        if comp_key == "O2"
        else (terms.frac_reduced, terms.frac_oxidised)
    )
    conversion, unreacted = _bounded(conversion), _bounded(unreacted)
    diffusivity = D_expr(comp_key, temperature_k=terms.temperature_k, conversion=conversion)
    gas_fraction = _available(symbols.concentration(comp_key) / terms.summed_gas_conc_molm3)
    concentration_power = terms.total_gas_conc_molm3**params.order * _rational_power(
        params.order, gas_fraction
    )
    one_third = _rational_power(ONE_THIRD, unreacted)
    two_thirds = _rational_power(TWO_THIRDS, unreacted)
    kinetic = k_expr(comp_key, temperature_k=terms.temperature_k)
    numerator = (
        3
        * params.b
        * two_thirds
        * kinetic
        * diffusivity
        * concentration_power**2
        / (params.r0 * params.cs)
    )
    denominator = concentration_power * (
        diffusivity + params.r0 * kinetic * (one_third - two_thirds)
    )
    conversion_rate = (
        _gate(gas_fraction, Y_GATE)
        * _gate(unreacted, F_GATE)
        * numerator
        / sp.Max(denominator, DENOMINATOR_FLOOR)
    )
    return terms.total_solid_inventory_molm3 * conversion_rate


def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    shared = _family_terms(symbols)
    specs = {
        "reduction_h2": ({"H2": 1, "NiO": 1}, {"Ni": 1, "H2O": 1}, "H2"),
        "reduction_co": ({"CO": 1, "NiO": 1}, {"Ni": 1, "CO2": 1}, "CO"),
        "oxidation_o2": ({"O2": sp.Rational(1, 2), "Ni": 1}, {"NiO": 1}, "O2"),
    }
    return {
        name: Reaction(
            f"medrano.{name}", reactants, products, (), _reaction_rate(symbols, key, shared=shared)
        )
        for name, (reactants, products, key) in specs.items()
    }
