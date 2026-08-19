"""Implementations in the Aye-plus-Bee-to-Cee reaction family."""

from collections.abc import Mapping

from sympy import Rational, exp

from ..model import CaseSymbols, Reaction

ACTIVATION_ENERGY = Rational(50_000)
GAS_CONSTANT = Rational("8.314")
RATE_CONSTANT = Rational(2)


def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    aye = symbols.concentration("Aye")
    bee = symbols.concentration("Bee")
    activation_factor = exp(
        -ACTIVATION_ENERGY / (GAS_CONSTANT * symbols.temperature)
    )
    return {
        "half_order": Reaction(
            id="aye_plus_bee_to_cee.half_order",
            reactants={"Aye": 1, "Bee": 1},
            products={"Cee": 1},
            catalysts=(),
            rate=RATE_CONSTANT * activation_factor * aye * bee ** Rational(1, 2),
        )
    }
