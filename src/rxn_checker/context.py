"""Typed, lazily constructed objects shared by one analysis run."""

from dataclasses import dataclass
from functools import cached_property

import sympy as sp

from .case import Case
from .domain import ConcentrationDomain, DomainKind
from .model import Species
from .network import ReactionNetwork, build_network
from .proof import ExpressionAnalyzer


@dataclass
class AnalysisContext:
    case: Case

    @cached_property
    def species_by_id(self) -> dict[str, Species]:
        return {species.id: species for species in self.case.species}

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

    @cached_property
    def expression_analyzer(self) -> ExpressionAnalyzer:
        return ExpressionAnalyzer()

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

__all__ = ("AnalysisContext",)
