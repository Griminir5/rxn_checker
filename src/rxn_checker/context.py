"""Typed, lazily constructed objects shared by one analysis run."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import sympy as sp

from .case import Case
from .domain import ConcentrationDomain, DomainKind
from .model import Reaction
from .network import ReactionNetwork, build_network
from .species import PROPERTY_REGISTRY, PropertyRegistry

if TYPE_CHECKING:
    from .checks.lipschitz_continuity import LipschitzContinuityResult


@dataclass
class AnalysisContext:
    case: Case
    property_registry: PropertyRegistry = PROPERTY_REGISTRY
    _rate_facts: dict[tuple[int, str], "LipschitzContinuityResult"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @cached_property
    def physical_domain(self) -> ConcentrationDomain:
        return self.case.domain_spec.build(DomainKind.PHYSICAL)

    @cached_property
    def augmented_domain(self) -> ConcentrationDomain:
        return self.case.domain_spec.build(DomainKind.AUGMENTED)

    def domain(self, kind: DomainKind) -> ConcentrationDomain:
        return (
            self.physical_domain
            if DomainKind(kind) is DomainKind.PHYSICAL
            else self.augmented_domain
        )

    @cached_property
    def network(self) -> ReactionNetwork:
        return build_network(self.case)

    @property
    def stoichiometry(self) -> sp.ImmutableMatrix:
        return self.network.stoichiometry

    def source_contributions(
        self,
        species_id: str,
    ) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
        """Return sparse ``(coefficient, rate)`` pairs for one species."""

        row = self.case.symbols.species_ids.index(species_id)
        return tuple(
            (coefficient, reaction.rate)
            for coefficient, reaction in zip(
                self.stoichiometry.row(row), self.case.reactions
            )
            if coefficient != 0
        )

    def rate_facts(
        self,
        reaction: Reaction,
        domain: ConcentrationDomain,
    ) -> "LipschitzContinuityResult":
        """Return one cached legacy rate analysis for a rate/domain pair."""

        key = id(domain), reaction.id
        if key not in self._rate_facts:
            from .checks.lipschitz_continuity import check_lipschitz_continuity

            self._rate_facts[key] = check_lipschitz_continuity(reaction, domain)
        return self._rate_facts[key]


__all__ = ("AnalysisContext",)
