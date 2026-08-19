from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import sympy as sp
from sympy import Symbol


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class IdealGasClosure:
    """Ideal-gas relationship for a case's gas concentrations."""

    gas_concentrations: tuple[Symbol, ...]
    temperature: Symbol
    pressure: Symbol
    gas_constant: float = GAS_CONSTANT_J_PER_MOL_K
    minimum_total: str = "positive"

    def __post_init__(self) -> None:
        gas_concentrations = tuple(self.gas_concentrations)
        if not gas_concentrations:
            raise ValueError("Ideal-gas closure requires at least one gas species.")
        if len(gas_concentrations) != len(set(gas_concentrations)):
            raise ValueError("Ideal-gas closure gas concentrations must be unique.")
        if not math.isfinite(self.gas_constant) or self.gas_constant <= 0:
            raise ValueError("Ideal-gas constant must be finite and positive.")
        if self.minimum_total not in {"positive", "ideal_gas"}:
            raise ValueError(
                "Ideal-gas minimum total must be 'positive' or 'ideal_gas'."
            )
        object.__setattr__(self, "gas_concentrations", gas_concentrations)

    @property
    def total_concentration(self) -> sp.Expr:
        return sp.Add(*self.gas_concentrations)

    @property
    def equation(self) -> sp.Equality:
        """Return ``P = c_total R T`` without evaluating the equality."""

        return sp.Eq(
            self.pressure,
            self.total_concentration * self.gas_constant * self.temperature,
            evaluate=False,
        )

    @property
    def augmented_strict_constraints(self) -> tuple[sp.Expr, ...]:
        """Linear consequences used by the augmented-domain prover."""

        return (self.total_concentration,) if self.minimum_total == "positive" else ()

    def derived_minimum_total(
        self,
        state_bounds: Mapping[Symbol, "VariableBounds"],
    ) -> sp.Expr | None:
        """Return ``P_min / (R T_max)`` when selected by the case."""

        if self.minimum_total != "ideal_gas":
            return None
        pressure_minimum = sp.Rational(
            str(state_bounds[self.pressure].physical_lower)
        )
        temperature_maximum = sp.Rational(
            str(state_bounds[self.temperature].physical_upper)
        )
        gas_constant = sp.Rational(str(self.gas_constant))
        return pressure_minimum / (gas_constant * temperature_maximum)


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
