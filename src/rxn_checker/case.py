"""Validated reaction case."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .domain import DomainSpec
from .model import CaseSymbols, Phase, Reaction, Species


@dataclass(frozen=True)
class Case:
    name: str
    species: tuple[Species, ...]
    symbols: CaseSymbols
    reactions: tuple[Reaction, ...]
    domain_spec: DomainSpec
    inert_species: tuple[str, ...] = ()
    check_config: Mapping[str, object] = field(default_factory=dict)
    report_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        species, reactions, inerts = (
            tuple(self.species),
            tuple(self.reactions),
            tuple(self.inert_species),
        )
        ids = tuple(item.id for item in species)
        if not self.name:
            raise ValueError("Case name must not be empty.")
        if len(ids) != len(set(ids)):
            raise ValueError("Case species must not contain duplicates.")
        if ids != self.symbols.species_ids:
            raise ValueError("Case species and symbols do not match.")
        if self.domain_spec.symbols != self.symbols:
            raise ValueError("Case and domain symbols differ.")
        reaction_ids = tuple(item.id for item in reactions)
        if not reactions:
            raise ValueError("Case must contain at least one reaction.")
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Reaction ids must be unique.")
        if len(inerts) != len(set(inerts)) or set(inerts) - set(ids):
            raise ValueError("Case inert species must be unique, known species.")
        for reaction in reactions:
            missing = set(reaction.species_ids) - set(ids)
            if missing:
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: "
                    + ", ".join(sorted(missing))
                    + "."
                )
            if set(reaction.species_ids) & set(inerts):
                raise ValueError(f"Inert species participate in reaction '{reaction.id}'.")
            unknown = reaction.rate.free_symbols - self.symbols.all_symbols
            if unknown:
                raise ValueError(
                    f"Reaction '{reaction.id}' uses symbols outside this case: "
                    + ", ".join(sorted(map(str, unknown)))
                    + "."
                )
        phases = {self.symbols.concentration(item.id): item.phase for item in species}
        for total in self.domain_spec.total_constraints:
            expected = Phase(total.name)
            if any(phases[symbol] is not expected for symbol in total.symbols):
                raise ValueError(f"Domain {total.name} total includes another phase.")
        for key, value in (
            ("species", species),
            ("reactions", reactions),
            ("inert_species", inerts),
            ("check_config", dict(self.check_config)),
            ("report_config", dict(self.report_config)),
        ):
            object.__setattr__(self, key, value)
