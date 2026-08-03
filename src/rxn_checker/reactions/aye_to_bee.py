"""Implementations in the Aye-to-Bee reaction family."""

from ..reaction import Reaction
from ..state import StateVariables

AUTOCATALYTIC_RATE_CONSTANT = 2.0
SIMPLE_RATE_CONSTANT = 2.0


def build_autocatalytic(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    bee = states.concentration("Bee")
    return Reaction(
        name="autocatalytic",
        family="aye_to_bee",
        reactants={"Aye": 1, "Bee": 1},
        products={"Bee": 2},
        rate=AUTOCATALYTIC_RATE_CONSTANT * aye * bee,
    )


def build_simple(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    return Reaction(
        name="simple",
        family="aye_to_bee",
        reactants={"Aye": 1},
        products={"Bee": 1},
        rate=SIMPLE_RATE_CONSTANT * aye,
    )


REACTIONS = {
    "autocatalytic": build_autocatalytic,
    "simple": build_simple,
}
