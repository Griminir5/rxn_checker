"""Exact network expressions shared by checks and symbolic profiles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import gcd, lcm
from types import MappingProxyType

import sympy as sp

from .case import Case
from .model import Reaction


def primitive_integer_vector(
    vector: sp.MatrixBase,
) -> tuple[sp.Integer, ...]:
    """Scale a nonzero rational vector to canonical coprime integers."""

    values = tuple(sp.Rational(value) for value in vector)
    nonzero = tuple(value for value in values if value)
    if not nonzero:
        raise ValueError("A primitive vector must contain a nonzero value.")
    scale = lcm(*(int(sp.denom(value)) for value in values))
    integers = [int(value * scale) for value in values]
    divisor = gcd(*(abs(value) for value in integers if value))
    integers = [value // divisor for value in integers]
    if next(value for value in integers if value) < 0:
        integers = [-value for value in integers]
    return tuple(sp.Integer(value) for value in integers)


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


@dataclass(frozen=True)
class SourceFlux:
    """One source-equivalent rate and its canonical stoichiometric direction."""

    id: str
    expression: sp.Expr
    stoichiometry: tuple[sp.Integer, ...]
    members: tuple[tuple[str, sp.Rational], ...]


@dataclass(frozen=True)
class FluxNetwork:
    """Reaction network after proportional stoichiometric columns are grouped."""

    fluxes: tuple[SourceFlux, ...]
    stoichiometry: sp.ImmutableMatrix

    @cached_property
    def rates(self) -> sp.ImmutableMatrix:
        return sp.ImmutableMatrix([flux.expression for flux in self.fluxes])

    @cached_property
    def rank_factorization(
        self,
    ) -> tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]:
        basis, coordinates = self.stoichiometry.rank_decomposition()
        return sp.ImmutableMatrix(basis), sp.ImmutableMatrix(coordinates)

    @cached_property
    def basis_ids(self) -> tuple[str, ...]:
        pivots = self.stoichiometry.rref()[1]
        return tuple(self.fluxes[index].id for index in pivots)


def _flux_id(reactions: Sequence[Reaction], number: int) -> str:
    if len(reactions) == 1:
        return reactions[0].id
    families = {reaction.family for reaction in reactions}
    names = tuple(reaction.name for reaction in reactions)
    stems = tuple(name.removesuffix("_fw").removesuffix("_bw") for name in names)
    if (
        len(families) == 1
        and len(set(stems)) == 1
        and all(name.endswith(("_fw", "_bw")) for name in names)
    ):
        family = next(iter(families))
        return f"{family + '.' if family else ''}{stems[0]}_net"
    return f"source_flux_{number}"


def source_equivalent_fluxes(
    reactions: Sequence[Reaction],
    stoichiometry: sp.MatrixBase,
) -> FluxNetwork:
    """Group proportional columns without changing the source vector ``S r``."""

    reactions = tuple(reactions)
    if stoichiometry.cols != len(reactions):
        raise ValueError("Stoichiometry columns must match the reactions.")

    groups: dict[tuple[sp.Integer, ...], list[tuple[int, sp.Rational]]] = {}
    for column in range(stoichiometry.cols):
        vector = stoichiometry[:, column]
        primitive = primitive_integer_vector(vector)
        pivot = next(index for index, value in enumerate(primitive) if value)
        coefficient = sp.Rational(vector[pivot] / primitive[pivot])
        groups.setdefault(primitive, []).append((column, coefficient))

    fluxes = []
    for number, (primitive, members) in enumerate(groups.items(), 1):
        if members[0][1] < 0:
            primitive = tuple(-value for value in primitive)
            members = [(index, -coefficient) for index, coefficient in members]
        selected = tuple(reactions[index] for index, _ in members)
        terms = tuple(
            reaction.rate
            if coefficient == 1
            else sp.Mul(coefficient, reaction.rate, evaluate=False)
            for reaction, (_, coefficient) in zip(selected, members)
        )
        expression = terms[0] if len(terms) == 1 else sp.Add(*terms, evaluate=False)
        fluxes.append(
            SourceFlux(
                _flux_id(selected, number),
                expression,
                primitive,
                tuple(
                    (reaction.id, coefficient)
                    for reaction, (_, coefficient) in zip(selected, members)
                ),
            )
        )

    matrix = sp.ImmutableMatrix.hstack(
        *(sp.ImmutableMatrix(flux.stoichiometry) for flux in fluxes)
    )
    return FluxNetwork(tuple(fluxes), matrix)


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


__all__ = (
    "FluxNetwork",
    "ReactionNetwork",
    "SourceFlux",
    "build_network",
    "primitive_integer_vector",
    "source_equivalent_fluxes",
)
