from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from sympy import Symbol


@dataclass(frozen=True)
class StateVariables:
    """All symbolic state variables owned by one case."""

    species_ids: tuple[str, ...]
    concentrations: Mapping[str, Symbol] = field(init=False)
    temperature: Symbol = field(init=False)
    pressure: Symbol = field(init=False)

    def __post_init__(self) -> None:
        species_ids = tuple(self.species_ids)
        if any(
            not isinstance(species_id, str)
            or not species_id
            or species_id != species_id.strip()
            for species_id in species_ids
        ):
            raise ValueError("Case species ids must not be blank or padded.")
        if len(species_ids) != len(set(species_ids)):
            raise ValueError("Case species must not contain duplicates.")

        object.__setattr__(self, "species_ids", species_ids)
        object.__setattr__(
            self,
            "concentrations",
            MappingProxyType(
                {
                    species_id: Symbol(species_id, real=True)
                    for species_id in species_ids
                }
            ),
        )
        object.__setattr__(self, "temperature", Symbol("temperature", real=True))
        object.__setattr__(self, "pressure", Symbol("pressure", real=True))

    def concentration(self, species_id: str) -> Symbol:
        try:
            return self.concentrations[species_id]
        except KeyError as exc:
            raise KeyError(f"Case has no species '{species_id}'.") from exc

    @property
    def symbols(self) -> frozenset[Symbol]:
        return frozenset(
            (*self.concentrations.values(), self.temperature, self.pressure)
        )


__all__ = ("StateVariables",)
