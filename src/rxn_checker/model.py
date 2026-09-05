"""Exact core data model for reaction cases."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import sympy as sp

RationalInput = int | float | str | sp.Rational


def parse_rational(value: RationalInput, *, label="value") -> sp.Rational:
    """Parse a finite rational from decimal spelling, never binary float state."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite rational number.")
    try:
        result = sp.Rational(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a finite rational number.") from error
    if result.is_finite is not True:
        raise ValueError(f"{label} must be a finite rational number.")
    return result


def exact_expr(value) -> sp.Expr:
    expression = sp.sympify(value)
    return expression.xreplace({item: parse_rational(item) for item in expression.atoms(sp.Float)})


def _name(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must not be blank or padded.")


class Phase(StrEnum):
    GAS = "gas"
    SOLID = "solid"


@dataclass(frozen=True)
class Species:
    id: str
    name: str
    phase: Phase
    atoms: Mapping[str, RationalInput]
    molar_mass: RationalInput | None = None

    def __post_init__(self):
        _name(self.id, "Species id")
        _name(self.name, "Species name")
        if self.id in {"temperature", "pressure"}:
            raise ValueError(f"Species id '{self.id}' is reserved.")
        try:
            phase = Phase(self.phase)
        except (TypeError, ValueError) as error:
            raise ValueError("Species phase must be either 'gas' or 'solid'.") from error
        if not isinstance(self.atoms, Mapping) or not self.atoms:
            raise ValueError(f"Species '{self.id}' must define its atoms.")
        atoms = {}
        for element, value in self.atoms.items():
            _name(element, f"Element in species '{self.id}'")
            count = parse_rational(
                value, label=f"Atom count for '{element}' in species '{self.id}'"
            )
            if count <= 0:
                raise ValueError("Atom counts must be positive.")
            atoms[element] = count
        mass = (
            None
            if self.molar_mass is None
            else parse_rational(self.molar_mass, label=f"Molar mass for species '{self.id}'")
        )
        if mass is not None and mass <= 0:
            raise ValueError("Molar mass must be positive.")
        for key, value in (("phase", phase), ("atoms", atoms), ("molar_mass", mass)):
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class CaseSymbols:
    concentrations: Mapping[str, sp.Symbol]
    temperature: sp.Symbol
    pressure: sp.Symbol

    def __post_init__(self):
        concentrations = dict(self.concentrations)
        if not concentrations:
            raise ValueError("A case must declare at least one species.")
        for species_id in concentrations:
            _name(species_id, "Case species id")
        if set(concentrations) & {"temperature", "pressure"}:
            raise ValueError("Temperature and pressure are reserved names.")
        symbols = (*concentrations.values(), self.temperature, self.pressure)
        if any(not isinstance(symbol, sp.Symbol) for symbol in symbols):
            raise TypeError("Case symbols must be SymPy Symbol objects.")
        if any(symbol.is_real is not True for symbol in symbols):
            raise ValueError("Case symbols must be declared real.")
        if len(symbols) != len(set(symbols)):
            raise ValueError("Every case symbol must be distinct.")
        object.__setattr__(self, "concentrations", concentrations)

    @classmethod
    def for_species(cls, species_ids: Iterable[str]):
        ids = tuple(species_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("Case species must not contain duplicates.")
        return cls(
            {item: sp.Symbol(item, real=True) for item in ids},
            sp.Symbol("temperature", real=True),
            sp.Symbol("pressure", real=True),
        )

    def concentration(self, species_id):
        try:
            return self.concentrations[species_id]
        except KeyError as error:
            raise KeyError(f"Case has no species '{species_id}'.") from error

    @property
    def species_ids(self):
        return tuple(self.concentrations)

    @property
    def concentration_symbols(self):
        return frozenset(self.concentrations.values())

    @property
    def parameter_symbols(self):
        return frozenset((self.temperature, self.pressure))

    @property
    def all_symbols(self):
        return self.concentration_symbols | self.parameter_symbols


def _side(values, label):
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} coefficients must be a mapping.")
    result = {}
    for species_id, value in values.items():
        _name(species_id, f"{label} species id")
        coefficient = parse_rational(value, label=f"{label} coefficient for '{species_id}'")
        if coefficient <= 0:
            raise ValueError(f"{label} coefficients must be positive.")
        result[species_id] = coefficient
    return result


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: Mapping[str, RationalInput]
    products: Mapping[str, RationalInput]
    catalysts: tuple[str, ...]
    rate: sp.Expr
    net_stoichiometry: Mapping[str, sp.Rational] = field(init=False)

    def __post_init__(self):
        _name(self.id, "Reaction id")
        reactants, products, catalysts = (
            _side(self.reactants, "Reactant"),
            _side(self.products, "Product"),
            tuple(self.catalysts),
        )
        if not reactants and not products:
            raise ValueError(f"Reaction '{self.id}' must have a reactant or product.")
        for item in catalysts:
            _name(item, "Catalyst id")
        if len(catalysts) != len(set(catalysts)):
            raise ValueError("Catalysts must be unique.")
        if (set(reactants) | set(products)) & set(catalysts):
            raise ValueError("Catalysts must not also be reactants or products.")
        rate = sp.sympify(self.rate)
        if not isinstance(rate, sp.Expr):
            raise TypeError("Reaction rate must be a scalar SymPy expression.")
        if rate.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I) or any(
            number.is_finite is False or number.is_real is False for number in rate.atoms(sp.Number)
        ):
            raise ValueError(
                f"Reaction '{self.id}' rate contains a non-finite or complex constant."
            )
        net = {
            item: products.get(item, 0) - reactants.get(item, 0)
            for item in dict.fromkeys((*reactants, *products))
        }
        net = {item: value for item, value in net.items() if value}
        if not net:
            raise ValueError(f"Reaction '{self.id}' has zero net stoichiometry.")
        for key, value in (
            ("reactants", reactants),
            ("products", products),
            ("catalysts", catalysts),
            ("rate", rate),
            ("net_stoichiometry", net),
        ):
            object.__setattr__(self, key, value)

    @property
    def family(self):
        return self.id.rpartition(".")[0]

    @property
    def name(self):
        return self.id.rsplit(".", 1)[-1]

    @property
    def species_ids(self):
        return tuple(dict.fromkeys((*self.reactants, *self.products, *self.catalysts)))
