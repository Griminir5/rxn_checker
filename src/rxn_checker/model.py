"""Exact core data model for reaction cases."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import sympy as sp


RationalInput = int | float | str | sp.Rational


def parse_rational(value: RationalInput, *, label: str = "value") -> sp.Rational:
    """Parse a finite rational from its decimal spelling, never via binary floats."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite rational number.")
    try:
        parsed = sp.Rational(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a finite rational number.") from error
    if parsed.is_finite is not True:
        raise ValueError(f"{label} must be a finite rational number.")
    return parsed


class Phase(StrEnum):
    """Supported material phases."""

    GAS = "gas"
    SOLID = "solid"


@dataclass(frozen=True)
class Species:
    """One immutable species definition using exact rational quantities."""

    id: str
    name: str
    phase: Phase
    atoms: Mapping[str, RationalInput]
    molar_mass: RationalInput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id != self.id.strip():
            raise ValueError("Species id must not be blank or padded.")
        if self.id in {"temperature", "pressure"}:
            raise ValueError(f"Species id '{self.id}' is reserved.")
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("Species name must not be blank or padded.")
        try:
            phase = Phase(self.phase)
        except (TypeError, ValueError) as error:
            raise ValueError("Species phase must be either 'gas' or 'solid'.") from error

        if not isinstance(self.atoms, Mapping) or not self.atoms:
            raise ValueError(f"Species '{self.id}' must define its atoms.")
        atoms: dict[str, sp.Rational] = {}
        for element, value in self.atoms.items():
            if (
                not isinstance(element, str)
                or not element
                or element != element.strip()
            ):
                raise ValueError(f"Species '{self.id}' has an invalid element name.")
            count = parse_rational(
                value,
                label=f"Atom count for '{element}' in species '{self.id}'",
            )
            if count <= 0:
                raise ValueError(
                    f"Atom count for '{element}' in species '{self.id}' "
                    "must be positive."
                )
            atoms[element] = count

        molar_mass = self.molar_mass
        if molar_mass is not None:
            molar_mass = parse_rational(
                molar_mass,
                label=f"Molar mass for species '{self.id}'",
            )
            if molar_mass <= 0:
                raise ValueError(
                    f"Molar mass for species '{self.id}' must be positive."
                )

        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "atoms", MappingProxyType(atoms))
        object.__setattr__(self, "molar_mass", molar_mass)

    @property
    def mw(self) -> sp.Rational | None:
        """Compatibility name for the legacy chemistry checks."""

        return self.molar_mass


@dataclass(frozen=True)
class CaseSymbols:
    """Concentration coordinates and uniformly bounded external parameters."""

    concentrations: Mapping[str, sp.Symbol]
    temperature: sp.Symbol
    pressure: sp.Symbol

    def __post_init__(self) -> None:
        concentrations = dict(self.concentrations)
        if not concentrations:
            raise ValueError("A case must declare at least one species.")
        if any(
            not isinstance(species_id, str)
            or not species_id
            or species_id != species_id.strip()
            for species_id in concentrations
        ):
            raise ValueError("Case species ids must not be blank or padded.")
        reserved = set(concentrations) & {"temperature", "pressure"}
        if reserved:
            raise ValueError(
                "Reserved names cannot be used as species ids: "
                + ", ".join(sorted(reserved))
                + "."
            )

        all_symbols = (*concentrations.values(), self.temperature, self.pressure)
        if any(not isinstance(symbol, sp.Symbol) for symbol in all_symbols):
            raise TypeError("Case symbols must be SymPy Symbol objects.")
        if any(symbol.is_real is not True for symbol in all_symbols):
            raise ValueError("Case symbols must be declared real.")
        if len(all_symbols) != len(set(all_symbols)):
            raise ValueError("Every case symbol must be distinct.")

        object.__setattr__(self, "concentrations", MappingProxyType(concentrations))

    @classmethod
    def for_species(cls, species_ids: Iterable[str]) -> "CaseSymbols":
        """Construct the conventional real symbols for a sequence of species ids."""

        ids = tuple(species_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Case species must not contain duplicates.")
        concentrations = {
            species_id: sp.Symbol(species_id, real=True) for species_id in ids
        }
        return cls(
            concentrations=concentrations,
            temperature=sp.Symbol("temperature", real=True),
            pressure=sp.Symbol("pressure", real=True),
        )

    def concentration(self, species_id: str) -> sp.Symbol:
        try:
            return self.concentrations[species_id]
        except KeyError as error:
            raise KeyError(f"Case has no species '{species_id}'.") from error

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(self.concentrations)

    @property
    def concentration_symbols(self) -> frozenset[sp.Symbol]:
        return frozenset(self.concentrations.values())

    @property
    def parameter_symbols(self) -> frozenset[sp.Symbol]:
        return frozenset((self.temperature, self.pressure))

    @property
    def all_symbols(self) -> frozenset[sp.Symbol]:
        return self.concentration_symbols | self.parameter_symbols

    @property
    def symbols(self) -> frozenset[sp.Symbol]:
        """Compatibility view used by checks that predate ``CaseSymbols``."""

        return self.all_symbols


def _reaction_side(
    values: Mapping[str, RationalInput],
    label: str,
) -> Mapping[str, sp.Rational]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} coefficients must be a mapping.")
    side: dict[str, sp.Rational] = {}
    for species_id, value in values.items():
        if (
            not isinstance(species_id, str)
            or not species_id
            or species_id != species_id.strip()
        ):
            raise ValueError(f"{label} species ids must not be blank or padded.")
        coefficient = parse_rational(
            value,
            label=f"{label} coefficient for '{species_id}'",
        )
        if coefficient <= 0:
            raise ValueError(f"{label} coefficients must be positive.")
        side[species_id] = coefficient
    return MappingProxyType(side)


@dataclass(frozen=True)
class Reaction:
    """One directional rate law with exact stoichiometric coefficients."""

    id: str
    reactants: Mapping[str, RationalInput]
    products: Mapping[str, RationalInput]
    catalysts: tuple[str, ...]
    rate: sp.Expr
    net_stoichiometry: Mapping[str, sp.Rational] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id != self.id.strip():
            raise ValueError("Reaction id must not be blank or padded.")

        reactants = _reaction_side(self.reactants, "Reactant")
        products = _reaction_side(self.products, "Product")
        if not reactants and not products:
            raise ValueError(f"Reaction '{self.id}' must have a reactant or product.")

        catalysts = tuple(self.catalysts)
        if any(
            not isinstance(species_id, str)
            or not species_id
            or species_id != species_id.strip()
            for species_id in catalysts
        ):
            raise ValueError("Catalyst ids must not be blank or padded.")
        if len(catalysts) != len(set(catalysts)):
            raise ValueError("Catalysts must be unique.")
        if (set(reactants) | set(products)) & set(catalysts):
            raise ValueError("Catalysts must not also be reactants or products.")

        rate = sp.sympify(self.rate)
        if not isinstance(rate, sp.Expr):
            raise TypeError("Reaction rate must be a scalar SymPy expression.")
        forbidden = (sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I)
        if rate.has(*forbidden) or any(
            number.is_finite is False or number.is_real is False
            for number in rate.atoms(sp.Number)
        ):
            raise ValueError(
                f"Reaction '{self.id}' rate contains a non-finite or complex constant."
            )

        species_ids = dict.fromkeys((*reactants, *products))
        net = {
            species_id: products.get(species_id, sp.S.Zero)
            - reactants.get(species_id, sp.S.Zero)
            for species_id in species_ids
        }
        net = {species_id: value for species_id, value in net.items() if value != 0}
        if not net:
            raise ValueError(
                f"Reaction '{self.id}' has an identically zero net stoichiometry."
            )

        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "catalysts", catalysts)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "net_stoichiometry", MappingProxyType(net))

    @property
    def family(self) -> str:
        return self.id.rsplit(".", 1)[0] if "." in self.id else ""

    @property
    def name(self) -> str:
        return self.id.rsplit(".", 1)[-1]

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.reactants, *self.products, *self.catalysts)))
