"""Shared reaction-network expressions."""

from collections.abc import Mapping
from types import MappingProxyType

import sympy as sp

from ..case import Case


def source_terms(case: Case) -> Mapping[str, sp.Expr]:
    """Return ``F = S r`` without algebraic rewriting."""

    return MappingProxyType(
        {
            species_id: sp.Add(
                *(
                    sp.Rational(str(reaction.net_stoichiometry.get(species_id, 0)))
                    * reaction.rate
                    for reaction in case.reactions
                )
            )
            for species_id in case.states.species_ids
        }
    )
