"""Exact network expressions shared by checks."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import sympy as sp

from .case import Case


@dataclass(frozen=True)
class ReactionNetwork:
    stoichiometry: sp.ImmutableMatrix
    rates: sp.ImmutableMatrix
    species_ids: tuple[str, ...]

    @cached_property
    def source_vector(self) -> sp.ImmutableMatrix:
        """Expand ``S r`` only for analyses that explicitly request it."""

        return sp.ImmutableMatrix(self.stoichiometry * self.rates)

    @cached_property
    def source_terms(self) -> Mapping[str, sp.Expr]:
        source = self.source_vector
        return MappingProxyType(
            {
                species_id: source[row]
                for row, species_id in enumerate(self.species_ids)
            }
        )


def build_network(case: Case) -> ReactionNetwork:
    matrix = sp.ImmutableMatrix(
        [
            [
                reaction.net_stoichiometry.get(species_id, sp.S.Zero)
                for reaction in case.reactions
            ]
            for species_id in case.symbols.species_ids
        ]
    )
    rates = sp.ImmutableMatrix([reaction.rate for reaction in case.reactions])
    return ReactionNetwork(matrix, rates, case.symbols.species_ids)


__all__ = ("ReactionNetwork", "build_network")
