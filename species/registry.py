from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class SpeciesProperties:
    """Property data for one species, stored in canonical SI units."""

    name: str
    mw: float | None = None
    atoms: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.mw is not None and (not math.isfinite(self.mw) or self.mw <= 0.0):
            raise ValueError(
                f"Molecular weight must be finite and positive for species '{self.name}'."
            )


@dataclass(frozen=True)
class PropertyRegistry:
    records: Mapping[str, SpeciesProperties]

    def __post_init__(self) -> None:
        for species_id, record in self.records.items():
            if not species_id or species_id != species_id.strip():
                raise ValueError(
                    "Property registry species identifiers must not be blank or padded."
                )

    def has_species(self, species_id: str) -> bool:
        return species_id in self.records

    def get_record(self, species_id: str) -> SpeciesProperties:
        try:
            return self.records[species_id]
        except KeyError as exc:
            available = ", ".join(self.records.keys())
            raise KeyError(
                f"Unknown species '{species_id}'. Available species: {available}"
            ) from exc


PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "Ar": SpeciesProperties(
            "Argon",
            39.948e-3,
        ),
        "CH4": SpeciesProperties(
            "Methane",
            16.043e-3,
        ),
        "CO": SpeciesProperties(
            "Carbon Monoxide",
            28.010e-3,
        ),
        "CO2": SpeciesProperties(
            "Carbon Dioxide",
            44.0095e-3,
        ),
        "H2": SpeciesProperties(
            "Hydrogen",
            2.01588e-3,
        ),
        "H2O": SpeciesProperties(
            "Water",
            18.01528e-3,
        ),
        "He": SpeciesProperties(
            "Helium",
            4.002602e-3,
        ),
        "N2": SpeciesProperties(
            "Nitrogen",
            28.0134e-3,
        ),
        "O2": SpeciesProperties(
            "Oxygen",
            31.9988e-3,
        ),
        "Ni": SpeciesProperties(
            "Nickel",
            58.693e-3,
        ),
        "NiO": SpeciesProperties(
            "Nickel Oxide",
            74.6928e-3,
        ),
        "CaAl2O4": SpeciesProperties(
            "Calcium Aluminate",
            158.039e-3,
        ),
        "Cu": SpeciesProperties(
            "Copper",
            63.55e-3,
        ),
        "Cu2O": SpeciesProperties(
            "Copper(I) Oxide",
            143.091e-3,
        ),
        "CuO": SpeciesProperties(
            "Copper(II) Oxide",
            79.545e-3,
        ),
        "Al2O3": SpeciesProperties(
            "Aluminium Oxide",
            101.961e-3,
        ),
        "CuAlO2": SpeciesProperties(
            "Copper(I) Aluminate",
            122.526e-3,
        ),
        "CuAl2O4": SpeciesProperties(
            "Copper(II) Aluminate",
            181.508e-3,
        ),
        "Fe": SpeciesProperties(
            "Iron",
            55.845e-3,
        ),
        "FeO": SpeciesProperties(
            "Iron(II) Oxide",
            71.844e-3,
        ),
        "Fe3O4": SpeciesProperties(
            "Iron(II,III) Oxide",
            231.533e-3,
        ),
        "Fe2O3": SpeciesProperties(
            "Iron(III) Oxide",
            159.687e-3,
        ),
    }
)
