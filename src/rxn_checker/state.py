"""Transitional bound helpers for checks that predate the unified domain model."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import sympy as sp

from .model import CaseSymbols, RationalInput, parse_rational


GAS_CONSTANT_J_PER_MOL_K = sp.Rational("8.31446261815324")


@dataclass(frozen=True)
class VariableBounds:
    """Exact physical bounds and an optional lower concentration excursion."""

    physical_lower: RationalInput
    physical_upper: RationalInput
    excursion_lower: RationalInput | None = None
    strict_lower: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict_lower, bool):
            raise ValueError("Strict-lower flag must be boolean.")
        lower = parse_rational(self.physical_lower, label="Physical lower bound")
        upper = parse_rational(self.physical_upper, label="Physical upper bound")
        if lower >= upper:
            raise ValueError("Physical lower bound must be below the upper bound.")

        excursion = self.excursion_lower
        if excursion is not None:
            excursion = parse_rational(excursion, label="Excursion lower bound")
            if excursion > lower:
                raise ValueError(
                    "Excursion lower bound must not exceed the physical lower bound."
                )

        object.__setattr__(self, "physical_lower", lower)
        object.__setattr__(self, "physical_upper", upper)
        object.__setattr__(self, "excursion_lower", excursion)

    def interval(
        self,
        *,
        include_excursion: bool = False,
    ) -> tuple[sp.Rational, sp.Rational]:
        lower = self.physical_lower
        if include_excursion and self.excursion_lower is not None:
            lower = self.excursion_lower
        return lower, self.physical_upper


@dataclass(frozen=True)
class IdealGasClosure:
    """Legacy view of the gas-total constraint used by existing checks."""

    gas_concentrations: tuple[sp.Symbol, ...]
    temperature: sp.Symbol
    pressure: sp.Symbol
    minimum_total: str
    explicit_minimum: RationalInput | None = None
    gas_constant: RationalInput = GAS_CONSTANT_J_PER_MOL_K

    def __post_init__(self) -> None:
        gas_concentrations = tuple(self.gas_concentrations)
        if not gas_concentrations:
            raise ValueError("A gas-total constraint requires at least one gas species.")
        if len(gas_concentrations) != len(set(gas_concentrations)):
            raise ValueError("Gas-total concentration symbols must be unique.")
        if self.minimum_total not in {"positive", "ideal_gas", "explicit"}:
            raise ValueError(
                "Gas minimum total must be 'positive', 'ideal_gas', or 'explicit'."
            )

        gas_constant = parse_rational(self.gas_constant, label="Ideal-gas constant")
        if gas_constant <= 0:
            raise ValueError("Ideal-gas constant must be positive.")
        explicit = self.explicit_minimum
        if self.minimum_total == "explicit":
            if explicit is None:
                raise ValueError("An explicit gas minimum requires a value.")
            explicit = parse_rational(explicit, label="Explicit gas minimum")
            if explicit <= 0:
                raise ValueError("Explicit gas minimum must be positive.")
        elif explicit is not None:
            raise ValueError("Explicit gas minimum is valid only in explicit mode.")

        object.__setattr__(self, "gas_concentrations", gas_concentrations)
        object.__setattr__(self, "gas_constant", gas_constant)
        object.__setattr__(self, "explicit_minimum", explicit)

    @property
    def total_concentration(self) -> sp.Expr:
        return sp.Add(*self.gas_concentrations)

    @property
    def equation(self) -> sp.Equality:
        return sp.Eq(
            self.pressure,
            self.total_concentration * self.gas_constant * self.temperature,
            evaluate=False,
        )

    @property
    def augmented_strict_constraints(self) -> tuple[sp.Expr, ...]:
        return (self.total_concentration,) if self.minimum_total == "positive" else ()

    def derived_minimum_total(
        self,
        state_bounds: Mapping[sp.Symbol, VariableBounds],
    ) -> sp.Expr | None:
        if self.minimum_total == "explicit":
            return self.explicit_minimum
        if self.minimum_total != "ideal_gas":
            return None
        pressure_minimum = state_bounds[self.pressure].physical_lower
        temperature_maximum = state_bounds[self.temperature].physical_upper
        return pressure_minimum / (self.gas_constant * temperature_maximum)


class StateVariables(CaseSymbols):
    """Backward-compatible constructor; new code should use ``CaseSymbols``."""

    def __init__(self, species_ids: Iterable[str]) -> None:
        symbols = CaseSymbols.for_species(species_ids)
        super().__init__(
            concentrations=symbols.concentrations,
            temperature=symbols.temperature,
            pressure=symbols.pressure,
        )


__all__ = (
    "GAS_CONSTANT_J_PER_MOL_K",
    "IdealGasClosure",
    "StateVariables",
    "VariableBounds",
)
