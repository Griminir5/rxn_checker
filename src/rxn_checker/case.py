"""Validated reaction case."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .domain import DomainSpec
from .model import CaseSymbols, Phase, Reaction, Species


@dataclass(frozen=True)
class Case:
    """An exact reaction system and the one specification for its domains."""

    name: str
    species: tuple[Species, ...]
    symbols: CaseSymbols
    reactions: tuple[Reaction, ...]
    domain_spec: DomainSpec
    inert_species: tuple[str, ...] = ()
    check_config: Mapping[str, object] = field(default_factory=dict)
    report_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        species = tuple(self.species)
        species_ids = tuple(item.id for item in species)
        if not self.name:
            raise ValueError("Case name must not be empty.")
        if len(species_ids) != len(set(species_ids)):
            raise ValueError("Case species must not contain duplicates.")
        if species_ids != self.symbols.species_ids:
            raise ValueError("Case species and concentration symbols do not match.")
        if self.domain_spec.symbols != self.symbols:
            raise ValueError("Case and domain specification use different symbols.")

        reactions = tuple(self.reactions)
        reaction_ids = tuple(reaction.id for reaction in reactions)
        if not reactions:
            raise ValueError("Case must contain at least one reaction.")
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Reaction ids must be unique within a case.")

        inerts = tuple(self.inert_species)
        if len(inerts) != len(set(inerts)):
            raise ValueError("Case inert species must not contain duplicates.")
        available = set(species_ids)
        unknown_inerts = set(inerts) - available
        if unknown_inerts:
            raise ValueError(
                "Case references unknown inert species: "
                + ", ".join(sorted(unknown_inerts))
                + "."
            )

        for reaction in reactions:
            missing = set(reaction.species_ids) - available
            if missing:
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: "
                    + ", ".join(sorted(missing))
                    + "."
                )
            participating_inerts = set(reaction.species_ids) & set(inerts)
            if participating_inerts:
                raise ValueError(
                    f"Inert species participate in reaction '{reaction.id}': "
                    + ", ".join(sorted(participating_inerts))
                    + "."
                )
            unknown_symbols = reaction.rate.free_symbols - self.symbols.all_symbols
            if unknown_symbols:
                raise ValueError(
                    f"Reaction '{reaction.id}' uses symbols not owned by this case: "
                    + ", ".join(sorted(map(str, unknown_symbols)))
                    + "."
                )

        phases = {
            self.symbols.concentration(item.id): item.phase for item in species
        }
        for constraint in self.domain_spec.total_constraints:
            try:
                expected = Phase(constraint.name)
            except ValueError as error:
                raise ValueError(
                    "Total constraints must be named 'gas' or 'solid'."
                ) from error
            wrong_phase = {
                str(symbol)
                for symbol in constraint.symbols
                if phases[symbol] is not expected
            }
            if wrong_phase:
                raise ValueError(
                    f"Domain {constraint.name} total includes another phase: "
                    + ", ".join(sorted(wrong_phase))
                    + "."
                )

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "inert_species", inerts)
        object.__setattr__(self, "check_config", dict(self.check_config))
        object.__setattr__(self, "report_config", dict(self.report_config))


__all__ = ("Case",)
