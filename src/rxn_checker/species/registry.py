from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpeciesProperties:
    """Properties of one species, with molecular weight in kg/mol."""

    name: str
    phase: str
    atoms: Mapping[str, float]
    mw: float | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Species name must not be blank or padded.")
        if self.phase not in {"gas", "solid"}:
            raise ValueError("Species phase must be either 'gas' or 'solid'.")
        if not self.atoms:
            raise ValueError(f"Species '{self.name}' must define its atoms.")
        if self.mw is not None and (not math.isfinite(self.mw) or self.mw <= 0):
            raise ValueError(
                f"Molecular weight must be finite and positive for species "
                f"'{self.name}'."
            )
        for element, count in self.atoms.items():
            if not element or element != element.strip():
                raise ValueError(f"Species '{self.name}' has an invalid element name.")
            if not math.isfinite(count) or count <= 0:
                raise ValueError(
                    f"Atom count for '{element}' in species '{self.name}' "
                    "must be finite and positive."
                )


@dataclass(frozen=True)
class PropertyRegistry:
    records: Mapping[str, SpeciesProperties]

    def __post_init__(self) -> None:
        for species_id in self.records:
            if not species_id or species_id != species_id.strip():
                raise ValueError(
                    "Property registry species identifiers must not be blank or padded."
                )

    def get_record(self, species_id: str) -> SpeciesProperties:
        try:
            return self.records[species_id]
        except KeyError as exc:
            available = ", ".join(self.records)
            raise KeyError(
                f"Unknown species '{species_id}'. Available species: {available}"
            ) from exc


PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "Aye": SpeciesProperties("Aye", "gas", {"Ex": 1}, 10.0e-3),
        "Bee": SpeciesProperties("Bee", "gas", {"Ex": 1}, 10.0e-3),
        "Cee": SpeciesProperties("Cee", "gas", {"Ex": 2}, 20.0e-3),
        "Ar": SpeciesProperties("Argon", "gas", {"Ar": 1}, 39.948e-3),
        "CH4": SpeciesProperties("Methane", "gas", {"C": 1, "H": 4}, 16.043e-3),
        "CO": SpeciesProperties("Carbon Monoxide", "gas", {"C": 1, "O": 1}, 28.010e-3),
        "CO2": SpeciesProperties("Carbon Dioxide", "gas", {"C": 1, "O": 2}, 44.0095e-3),
        "H2": SpeciesProperties("Hydrogen", "gas", {"H": 2}, 2.01588e-3),
        "H2O": SpeciesProperties("Water", "gas", {"H": 2, "O": 1}, 18.01528e-3),
        "He": SpeciesProperties("Helium", "gas", {"He": 1}, 4.002602e-3),
        "N2": SpeciesProperties("Nitrogen", "gas", {"N": 2}, 28.0134e-3),
        "O2": SpeciesProperties("Oxygen", "gas", {"O": 2}, 31.9988e-3),
        "Ni": SpeciesProperties("Nickel", "solid", {"Ni": 1}, 58.693e-3),
        "NiO": SpeciesProperties(
            "Nickel Oxide", "solid", {"Ni": 1, "O": 1}, 74.6928e-3
        ),
        "CaAl2O4": SpeciesProperties(
            "Calcium Aluminate", "solid", {"Ca": 1, "Al": 2, "O": 4}, 158.039e-3
        ),
        "Cu": SpeciesProperties("Copper", "solid", {"Cu": 1}, 63.55e-3),
        "Cu2O": SpeciesProperties(
            "Copper(I) Oxide", "solid", {"Cu": 2, "O": 1}, 143.091e-3
        ),
        "CuO": SpeciesProperties(
            "Copper(II) Oxide", "solid", {"Cu": 1, "O": 1}, 79.545e-3
        ),
        "Al2O3": SpeciesProperties(
            "Aluminium Oxide", "solid", {"Al": 2, "O": 3}, 101.961e-3
        ),
        "CuAlO2": SpeciesProperties(
            "Copper(I) Aluminate", "solid", {"Cu": 1, "Al": 1, "O": 2}, 122.526e-3
        ),
        "CuAl2O4": SpeciesProperties(
            "Copper(II) Aluminate", "solid", {"Cu": 1, "Al": 2, "O": 4}, 181.508e-3
        ),
        "Fe": SpeciesProperties("Iron", "solid", {"Fe": 1}, 55.845e-3),
        "FeO": SpeciesProperties(
            "Iron(II) Oxide", "solid", {"Fe": 1, "O": 1}, 71.844e-3
        ),
        "Fe3O4": SpeciesProperties(
            "Iron(II,III) Oxide", "solid", {"Fe": 3, "O": 4}, 231.533e-3
        ),
        "Fe2O3": SpeciesProperties(
            "Iron(III) Oxide", "solid", {"Fe": 2, "O": 3}, 159.687e-3
        ),
    }
)
