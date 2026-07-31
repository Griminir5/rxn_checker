from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from sympy import Expr, sympify


def _side(values: Mapping[str, float], label: str) -> Mapping[str, float]:
    values = dict(values)
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(f"{label} coefficients must be finite and positive.")
    return MappingProxyType(values)


@dataclass(frozen=True)
class Reaction:
    """One forward reaction and its symbolic rate expression."""

    id: str
    family: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate: Expr
    catalysts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.family:
            raise ValueError("Reaction id and family must not be empty.")

        reactants = _side(self.reactants, "Reactant")
        products = _side(self.products, "Product")
        catalysts = tuple(self.catalysts)
        if not reactants and not products:
            raise ValueError(f"Reaction '{self.id}' must have a reactant or product.")
        if (set(reactants) | set(products)) & set(catalysts):
            raise ValueError("Catalysts must not also be reactants or products.")

        rate = sympify(self.rate)
        if not isinstance(rate, Expr):
            raise TypeError("Reaction rate must be a scalar SymPy expression.")

        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "catalysts", catalysts)
        object.__setattr__(self, "rate", rate)

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.reactants, *self.products, *self.catalysts)))

    @property
    def net_stoichiometry(self) -> Mapping[str, float]:
        species_ids = dict.fromkeys((*self.reactants, *self.products))
        coefficients = {
            species_id: self.products.get(species_id, 0)
            - self.reactants.get(species_id, 0)
            for species_id in species_ids
        }
        return MappingProxyType(
            {
                species_id: coefficient
                for species_id, coefficient in coefficients.items()
                if coefficient
            }
        )


__all__ = ("Reaction",)
