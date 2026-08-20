"""Built-in exact species definitions."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..model import Phase, Species


@dataclass(frozen=True)
class PropertyRegistry:
    """Explicit lookup table for species definitions."""

    records: Mapping[str, Species]

    def __post_init__(self) -> None:
        records = dict(self.records)
        for species_id, species in records.items():
            if species_id != species.id:
                raise ValueError(
                    f"Species registry key '{species_id}' does not match "
                    f"record id '{species.id}'."
                )
        object.__setattr__(self, "records", MappingProxyType(records))

    def get_record(self, species_id: str) -> Species:
        try:
            return self.records[species_id]
        except KeyError as error:
            available = ", ".join(self.records)
            raise KeyError(
                f"Unknown species '{species_id}'. Available species: {available}"
            ) from error


def _species(
    species_id: str,
    name: str,
    phase: Phase,
    atoms: Mapping[str, int | float | str],
    molar_mass: int | float | str | None,
) -> Species:
    return Species(species_id, name, phase, atoms, molar_mass)


# Molar masses use one atomic-weight basis throughout and are stored exactly as
# their declared decimal spellings in kg/mol.
PROPERTY_REGISTRY = PropertyRegistry(
    records={
        "Aye": _species("Aye", "Aye", Phase.GAS, {"Ex": 1}, "0.010"),
        "Bee": _species("Bee", "Bee", Phase.GAS, {"Ex": 1}, "0.010"),
        "Cee": _species("Cee", "Cee", Phase.GAS, {"Ex": 2}, "0.020"),
        "Ar": _species("Ar", "Argon", Phase.GAS, {"Ar": 1}, "0.039948"),
        "CH4": _species(
            "CH4", "Methane", Phase.GAS, {"C": 1, "H": 4}, "0.01604246"
        ),
        "CO": _species(
            "CO", "Carbon Monoxide", Phase.GAS, {"C": 1, "O": 1}, "0.0280101"
        ),
        "CO2": _species(
            "CO2", "Carbon Dioxide", Phase.GAS, {"C": 1, "O": 2}, "0.0440095"
        ),
        "H2": _species("H2", "Hydrogen", Phase.GAS, {"H": 2}, "0.00201588"),
        "H2O": _species(
            "H2O", "Water", Phase.GAS, {"H": 2, "O": 1}, "0.01801528"
        ),
        "He": _species("He", "Helium", Phase.GAS, {"He": 1}, "0.004002602"),
        "N2": _species("N2", "Nitrogen", Phase.GAS, {"N": 2}, "0.0280134"),
        "O2": _species("O2", "Oxygen", Phase.GAS, {"O": 2}, "0.0319988"),
        "Ni": _species("Ni", "Nickel", Phase.SOLID, {"Ni": 1}, "0.0586934"),
        "NiO": _species(
            "NiO", "Nickel Oxide", Phase.SOLID, {"Ni": 1, "O": 1}, "0.0746928"
        ),
        "CaAl2O4": _species(
            "CaAl2O4",
            "Calcium Aluminate",
            Phase.SOLID,
            {"Ca": 1, "Al": 2, "O": 4},
            "0.1580386772",
        ),
        "Cu": _species("Cu", "Copper", Phase.SOLID, {"Cu": 1}, "0.063546"),
        "Cu2O": _species(
            "Cu2O", "Copper(I) Oxide", Phase.SOLID, {"Cu": 2, "O": 1}, "0.1430914"
        ),
        "CuO": _species(
            "CuO", "Copper(II) Oxide", Phase.SOLID, {"Cu": 1, "O": 1}, "0.0795454"
        ),
        "Al2O3": _species(
            "Al2O3",
            "Aluminium Oxide",
            Phase.SOLID,
            {"Al": 2, "O": 3},
            "0.1019612772",
        ),
        "CuAlO2": _species(
            "CuAlO2",
            "Copper(I) Aluminate",
            Phase.SOLID,
            {"Cu": 1, "Al": 1, "O": 2},
            "0.1225263386",
        ),
        "CuAl2O4": _species(
            "CuAl2O4",
            "Copper(II) Aluminate",
            Phase.SOLID,
            {"Cu": 1, "Al": 2, "O": 4},
            "0.1815066772",
        ),
        "Fe": _species("Fe", "Iron", Phase.SOLID, {"Fe": 1}, "0.055845"),
        "FeO": _species(
            "FeO", "Iron(II) Oxide", Phase.SOLID, {"Fe": 1, "O": 1}, "0.0718444"
        ),
        "Fe0.94O": _species(
            "Fe0.94O",
            "Wüstite",
            Phase.SOLID,
            {"Fe": "0.94", "O": 1},
            "0.0684937",
        ),
        "Fe3O4": _species(
            "Fe3O4",
            "Iron(II,III) Oxide",
            Phase.SOLID,
            {"Fe": 3, "O": 4},
            "0.2315326",
        ),
        "Fe2O3": _species(
            "Fe2O3",
            "Iron(III) Oxide",
            Phase.SOLID,
            {"Fe": 2, "O": 3},
            "0.1596882",
        ),
    }
)

__all__ = (
    "PROPERTY_REGISTRY",
    "PropertyRegistry",
)
