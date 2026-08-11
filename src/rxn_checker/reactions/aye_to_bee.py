"""Implementations in the Aye-to-Bee reaction family."""

from sympy import Expr, exp

from ..reaction import Reaction
from ..state import StateVariables

# Toy Arrhenius parameters in SI molar units.
ACTIVATION_ENERGY = 50_000.0  # J/mol
AUTOCATALYTIC_RATE_CONSTANT = 2.0
GAS_CONSTANT = 8.314  # J/(mol K)
SIMPLE_RATE_CONSTANT = 2.0


def _activation_factor(states: StateVariables) -> Expr:
    return exp(-ACTIVATION_ENERGY / (GAS_CONSTANT * states.temperature))


def build_autocatalytic(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    bee = states.concentration("Bee")
    return Reaction(
        name="autocatalytic",
        family="aye_to_bee",
        reactants={"Aye": 1, "Bee": 1},
        products={"Bee": 2},
        rate=(AUTOCATALYTIC_RATE_CONSTANT * _activation_factor(states) * aye * bee),
    )


def build_simple(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    return Reaction(
        name="simple",
        family="aye_to_bee",
        reactants={"Aye": 1},
        products={"Bee": 1},
        rate=SIMPLE_RATE_CONSTANT * _activation_factor(states) * aye,
    )


REACTIONS = {
    "autocatalytic": build_autocatalytic,
    "simple": build_simple,
}
