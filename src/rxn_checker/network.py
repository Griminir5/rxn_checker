"""Exact network expressions shared by checks."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from .case import Case


@dataclass(frozen=True)
class ReactionNetwork:
    stoichiometry: sp.ImmutableMatrix
    rates: sp.ImmutableMatrix
    source_vector: sp.ImmutableMatrix
    source_terms: Mapping[str, sp.Expr]


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
    source = matrix * rates
    terms = MappingProxyType(
        {
            species_id: source[row]
            for row, species_id in enumerate(case.symbols.species_ids)
        }
    )
    return ReactionNetwork(matrix, rates, sp.ImmutableMatrix(source), terms)


__all__ = ("ReactionNetwork", "build_network")
