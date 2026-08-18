"""Shared reaction-network expressions."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from ..case import Case


@dataclass(frozen=True)
class NetworkExpressions:
    """Exact stoichiometry, rates, and source terms shared across checks."""

    stoichiometric_matrix: sp.ImmutableMatrix
    rates: sp.ImmutableMatrix
    source_vector: sp.ImmutableMatrix
    source_terms: Mapping[str, sp.Expr]


def network_expressions(case: Case) -> NetworkExpressions:
    """Build ``S``, ``r``, and ``F = S r`` exactly once for a case."""

    matrix = sp.ImmutableMatrix(
        [
            [
                sp.Rational(
                    str(reaction.net_stoichiometry.get(species_id, 0))
                )
                for reaction in case.reactions
            ]
            for species_id in case.states.species_ids
        ]
    )
    rates = sp.ImmutableMatrix([reaction.rate for reaction in case.reactions])
    source = sp.ImmutableMatrix(matrix * rates)
    terms = MappingProxyType(
        {
            species_id: source[row]
            for row, species_id in enumerate(case.states.species_ids)
        }
    )
    return NetworkExpressions(matrix, rates, source, terms)


def source_terms(case: Case) -> Mapping[str, sp.Expr]:
    """Return ``F = S r`` without algebraic rewriting."""

    return network_expressions(case).source_terms
