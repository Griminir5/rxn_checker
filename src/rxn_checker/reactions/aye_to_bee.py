"""Implementations in the Aye-to-Bee reaction family."""

from collections.abc import Mapping

from sympy import Expr, Rational, exp

from ..model import CaseSymbols, Reaction

# Toy Arrhenius parameters in SI molar units.
ACTIVATION_ENERGY = Rational(50_000)  # J/mol
AUTOCATALYTIC_RATE_CONSTANT = Rational(2)
GAS_CONSTANT = Rational("8.314")  # J/(mol K)
SIMPLE_RATE_CONSTANT = Rational(2)


def _activation_factor(symbols: CaseSymbols) -> Expr:
    return exp(-ACTIVATION_ENERGY / (GAS_CONSTANT * symbols.temperature))


def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    aye = symbols.concentration("Aye")
    bee = symbols.concentration("Bee")
    activation_factor = _activation_factor(symbols)
    return {
        "autocatalytic": Reaction(
            id="aye_to_bee.autocatalytic",
            reactants={"Aye": 1, "Bee": 1},
            products={"Bee": 2},
            catalysts=(),
            rate=AUTOCATALYTIC_RATE_CONSTANT * activation_factor * aye * bee,
        ),
        "simple": Reaction(
            id="aye_to_bee.simple",
            reactants={"Aye": 1},
            products={"Bee": 1},
            catalysts=(),
            rate=SIMPLE_RATE_CONSTANT * activation_factor * aye,
        ),
    }
