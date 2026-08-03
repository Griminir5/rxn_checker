"""Implementations in the Aye-plus-Bee-to-Cee reaction family."""

from sympy import exp

from ..reaction import Reaction
from ..state import StateVariables

ACTIVATION_ENERGY = 50_000.0
GAS_CONSTANT = 8.314
RATE_CONSTANT = 2.0


def build_simple(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    bee = states.concentration("Bee")
    activation_factor = exp(
        -ACTIVATION_ENERGY / (GAS_CONSTANT * states.temperature)
    )
    return Reaction(
        name="half_order",
        family="aye_plus_bee_to_cee",
        reactants={"Aye": 1, "Bee": 1},
        products={"Cee": 1},
        rate=RATE_CONSTANT * activation_factor * aye * bee**0.5,
    )


REACTIONS = {
    "half_order": build_simple,
}
