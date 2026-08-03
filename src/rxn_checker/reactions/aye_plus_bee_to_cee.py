from ..reaction import Reaction
from ..state import StateVariables

RATE_CONSTANT = 2.0



def build_simple(states: StateVariables) -> Reaction:
    aye = states.concentration("Aye")
    bee = states.concentration("Bee")
    return Reaction(
        name="half_order",
        family="aye_plus_bee_to_cee",
        reactants={"Aye": 1, "Bee": 1},
        products={"Cee": 1},
        rate=RATE_CONSTANT * aye * bee**0.5,
    )


REACTIONS = {
    "half_order": build_simple,
}