"""Built-in exact species definitions."""

from dataclasses import dataclass

from .model import Phase, Species


@dataclass(frozen=True)
class PropertyRegistry:
    records: dict[str, Species]

    def __post_init__(self):
        if any(key != value.id for key, value in self.records.items()):
            raise ValueError("Species registry keys must match record ids.")

    def get_record(self, species_id):
        try:
            return self.records[species_id]
        except KeyError as error:
            raise KeyError(
                f"Unknown species '{species_id}'. Available species: " + ", ".join(self.records)
            ) from error


G, S = Phase.GAS, Phase.SOLID
# id, name, phase, elemental composition, molar mass in kg/mol
_DATA = (
    ("Aye", "Aye", G, {"Ex": 1}, "0.010"),
    ("Bee", "Bee", G, {"Ex": 1}, "0.010"),
    ("Cee", "Cee", G, {"Ex": 2}, "0.020"),
    ("Ar", "Argon", G, {"Ar": 1}, "0.039948"),
    ("CH4", "Methane", G, {"C": 1, "H": 4}, "0.01604246"),
    ("CO", "Carbon Monoxide", G, {"C": 1, "O": 1}, "0.0280101"),
    ("CO2", "Carbon Dioxide", G, {"C": 1, "O": 2}, "0.0440095"),
    ("H2", "Hydrogen", G, {"H": 2}, "0.00201588"),
    ("H2O", "Water", G, {"H": 2, "O": 1}, "0.01801528"),
    ("He", "Helium", G, {"He": 1}, "0.004002602"),
    ("N2", "Nitrogen", G, {"N": 2}, "0.0280134"),
    ("O2", "Oxygen", G, {"O": 2}, "0.0319988"),
    ("Ni", "Nickel", S, {"Ni": 1}, "0.0586934"),
    ("NiO", "Nickel Oxide", S, {"Ni": 1, "O": 1}, "0.0746928"),
    ("CaAl2O4", "Calcium Aluminate", S, {"Ca": 1, "Al": 2, "O": 4}, "0.1580386772"),
    ("Cu", "Copper", S, {"Cu": 1}, "0.063546"),
    ("Cu2O", "Copper(I) Oxide", S, {"Cu": 2, "O": 1}, "0.1430914"),
    ("CuO", "Copper(II) Oxide", S, {"Cu": 1, "O": 1}, "0.0795454"),
    ("Al2O3", "Aluminium Oxide", S, {"Al": 2, "O": 3}, "0.1019612772"),
    ("CuAlO2", "Copper(I) Aluminate", S, {"Cu": 1, "Al": 1, "O": 2}, "0.1225263386"),
    ("CuAl2O4", "Copper(II) Aluminate", S, {"Cu": 1, "Al": 2, "O": 4}, "0.1815066772"),
    ("Fe", "Iron", S, {"Fe": 1}, "0.055845"),
    ("FeO", "Iron(II) Oxide", S, {"Fe": 1, "O": 1}, "0.0718444"),
    ("Fe0.94O", "Wüstite", S, {"Fe": "0.94", "O": 1}, "0.0684937"),
    ("Fe3O4", "Iron(II,III) Oxide", S, {"Fe": 3, "O": 4}, "0.2315326"),
    ("Fe2O3", "Iron(III) Oxide", S, {"Fe": 2, "O": 3}, "0.1596882"),
)
PROPERTY_REGISTRY = PropertyRegistry({row[0]: Species(*row) for row in _DATA})
