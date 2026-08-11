from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from sympy import Symbol


@dataclass(frozen=True)
class VariableBounds:
    """Physical bounds and an optional lower excursion limit for one state."""

    physical_lower: float
    physical_upper: float
    excursion_lower: float | None = None
    strict_lower: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict_lower, bool):
            raise ValueError("Strict-lower flag must be boolean.")
        if not math.isfinite(self.physical_lower) or not math.isfinite(
            self.physical_upper
        ):
            raise ValueError("Physical bounds must be finite.")
        if self.physical_lower >= self.physical_upper:
            raise ValueError("Physical lower bound must be below the upper bound.")
        if self.excursion_lower is not None:
            if not math.isfinite(self.excursion_lower):
                raise ValueError("Excursion lower bound must be finite.")
            if self.excursion_lower > self.physical_lower:
                raise ValueError(
                    "Excursion lower bound must not exceed the physical lower bound."
                )

    def interval(self, *, include_excursion: bool = False) -> tuple[float, float]:
        lower = self.physical_lower
        if include_excursion and self.excursion_lower is not None:
            lower = self.excursion_lower
        return lower, self.physical_upper


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
