"""Validated reaction-case configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import sympy as sp

from .model import (
    CaseSymbols,
    Phase,
    RationalInput,
    Reaction,
    Species,
    parse_rational,
)
from .state import IdealGasClosure, VariableBounds


@dataclass(frozen=True)
class ParameterRange:
    """One finite, closed parameter interval."""

    lower: RationalInput
    upper: RationalInput

    def __post_init__(self) -> None:
        lower = parse_rational(self.lower, label="Parameter lower bound")
        upper = parse_rational(self.upper, label="Parameter upper bound")
        if lower >= upper:
            raise ValueError("Parameter lower bound must be below its upper bound.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class ParameterBox:
    temperature: ParameterRange
    pressure: ParameterRange

    def __post_init__(self) -> None:
        if self.temperature.lower <= 0:
            raise ValueError("Temperature lower bound must be positive.")


class ConcentrationModel(StrEnum):
    INDEPENDENT = "independent"
    CHAMFERED = "chamfered"


class TotalMinimumMode(StrEnum):
    NONE = "none"
    EXPLICIT = "explicit"
    IDEAL_GAS_MINIMUM = "ideal_gas_minimum"


@dataclass(frozen=True)
class TotalMinimumConfig:
    """Parsed configuration for one phase-total lower constraint."""

    mode: TotalMinimumMode
    species: tuple[str, ...] = ()
    value: RationalInput | None = None

    def __post_init__(self) -> None:
        try:
            mode = TotalMinimumMode(self.mode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unknown total-minimum mode '{self.mode}'.") from error
        species = tuple(self.species)
        if len(species) != len(set(species)):
            raise ValueError("Total-minimum species must be unique.")

        value = self.value
        if mode is TotalMinimumMode.EXPLICIT:
            if value is None:
                raise ValueError("An explicit total minimum requires a value.")
            value = parse_rational(value, label="Explicit total minimum")
            if value <= 0:
                raise ValueError("Explicit total minimum must be positive.")
        elif value is not None:
            raise ValueError("A total value is valid only in explicit mode.")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class DomainConfig:
    """Exact parsed domain settings; Phase 2 turns these into domain objects."""

    concentration_model: ConcentrationModel
    upper: Mapping[str, RationalInput]
    excursion_lower: Mapping[str, RationalInput]
    gas_total: TotalMinimumConfig = field(
        default_factory=lambda: TotalMinimumConfig(TotalMinimumMode.NONE)
    )
    solid_total: TotalMinimumConfig = field(
        default_factory=lambda: TotalMinimumConfig(TotalMinimumMode.NONE)
    )

    def __post_init__(self) -> None:
        try:
            concentration_model = ConcentrationModel(self.concentration_model)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown concentration model '{self.concentration_model}'."
            ) from error

        upper = {
            species_id: parse_rational(value, label=f"Upper bound for '{species_id}'")
            for species_id, value in self.upper.items()
        }
        if any(value <= 0 for value in upper.values()):
            raise ValueError("Concentration upper bounds must be positive.")
        excursion = {
            species_id: parse_rational(
                value,
                label=f"Excursion lower bound for '{species_id}'",
            )
            for species_id, value in self.excursion_lower.items()
        }
        if any(value > 0 for value in excursion.values()):
            raise ValueError("Concentration excursion lower bounds must be non-positive.")
        if set(upper) != set(excursion):
            raise ValueError(
                "Upper and excursion bounds must cover the same case species."
            )
        if concentration_model is ConcentrationModel.INDEPENDENT and (
            self.gas_total.mode is not TotalMinimumMode.NONE
            or self.solid_total.mode is not TotalMinimumMode.NONE
        ):
            raise ValueError("Independent concentration domains cannot define totals.")

        object.__setattr__(self, "concentration_model", concentration_model)
        object.__setattr__(self, "upper", MappingProxyType(upper))
        object.__setattr__(self, "excursion_lower", MappingProxyType(excursion))


@dataclass(frozen=True)
class CheckConfig:
    profile: str = "physical"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    fail_fast: str = "stage"


@dataclass(frozen=True)
class ReportConfig:
    verbosity: str = "failures"
    format: str = "text"
    output: str | None = None


@dataclass(frozen=True)
class Case:
    """A validated exact reaction system plus its parsed case configuration."""

    name: str
    species: tuple[Species, ...]
    symbols: CaseSymbols
    reactions: tuple[Reaction, ...]
    parameters: ParameterBox
    domain: DomainConfig
    inert_species: tuple[str, ...] = ()
    checks: CheckConfig = field(default_factory=CheckConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    state_bounds: Mapping[sp.Symbol, VariableBounds] = field(init=False, repr=False)
    gas_closure: IdealGasClosure | None = field(init=False, repr=False)
    species_by_id: Mapping[str, Species] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Case name must not be empty.")
        species = tuple(self.species)
        species_ids = tuple(item.id for item in species)
        if len(species_ids) != len(set(species_ids)):
            raise ValueError("Case species must not contain duplicates.")
        if species_ids != self.symbols.species_ids:
            raise ValueError("Case species and concentration symbols do not match.")
        if set(self.domain.upper) != set(species_ids):
            raise ValueError("Domain bounds must cover every case species exactly once.")

        species_by_id = dict(zip(species_ids, species))
        for total, phase in (
            (self.domain.gas_total, Phase.GAS),
            (self.domain.solid_total, Phase.SOLID),
        ):
            unknown = set(total.species) - set(species_by_id)
            if unknown:
                raise ValueError(
                    f"Domain {phase.value} total references unknown species: "
                    + ", ".join(sorted(unknown))
                    + "."
                )
            wrong_phase = {
                species_id
                for species_id in total.species
                if species_by_id[species_id].phase is not phase
            }
            if wrong_phase:
                raise ValueError(
                    f"Domain {phase.value} total includes species from another phase: "
                    + ", ".join(sorted(wrong_phase))
                    + "."
                )
            if (
                phase is Phase.SOLID
                and total.mode is TotalMinimumMode.IDEAL_GAS_MINIMUM
            ):
                raise ValueError("Solid totals do not support ideal_gas_minimum mode.")
            if total.mode is TotalMinimumMode.EXPLICIT:
                available = sum(self.domain.upper[item] for item in total.species)
                if available < total.value:
                    raise ValueError(
                        f"Domain {phase.value} total minimum exceeds its upper bounds."
                    )

        reactions = tuple(self.reactions)
        if not reactions:
            raise ValueError("Case must contain at least one reaction.")
        reaction_ids = tuple(reaction.id for reaction in reactions)
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Reaction ids must be unique within a case.")

        inert_species = tuple(self.inert_species)
        if len(inert_species) != len(set(inert_species)):
            raise ValueError("Case inert species must not contain duplicates.")
        unknown_inerts = set(inert_species) - set(species_ids)
        if unknown_inerts:
            raise ValueError(
                "Case references unknown inert species: "
                + ", ".join(sorted(unknown_inerts))
                + "."
            )

        available_species = set(species_ids)
        for reaction in reactions:
            missing = set(reaction.species_ids) - available_species
            if missing:
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: "
                    + ", ".join(sorted(missing))
                    + "."
                )
            participating_inerts = set(reaction.species_ids) & set(inert_species)
            if participating_inerts:
                raise ValueError(
                    f"Inert species participate in reaction '{reaction.id}': "
                    + ", ".join(sorted(participating_inerts))
                    + "."
                )
            unknown_symbols = reaction.rate.free_symbols - self.symbols.all_symbols
            if unknown_symbols:
                raise ValueError(
                    f"Reaction '{reaction.id}' uses symbols not owned by this case: "
                    + ", ".join(sorted(map(str, unknown_symbols)))
                    + "."
                )

        by_id = MappingProxyType(species_by_id)
        bounds: dict[sp.Symbol, VariableBounds] = {
            self.symbols.temperature: VariableBounds(
                self.parameters.temperature.lower,
                self.parameters.temperature.upper,
            ),
            self.symbols.pressure: VariableBounds(
                self.parameters.pressure.lower,
                self.parameters.pressure.upper,
            ),
        }
        for species_id, symbol in self.symbols.concentrations.items():
            bounds[symbol] = VariableBounds(
                0,
                self.domain.upper[species_id],
                self.domain.excursion_lower[species_id],
            )

        gas_total = self.domain.gas_total
        gas_closure: IdealGasClosure | None = None
        if gas_total.mode is not TotalMinimumMode.NONE:
            gas_symbols = tuple(
                self.symbols.concentration(species_id)
                for species_id in gas_total.species
            )
            if gas_total.mode is TotalMinimumMode.IDEAL_GAS_MINIMUM:
                if self.parameters.pressure.lower <= 0:
                    raise ValueError(
                        "Ideal-gas minimum requires a positive pressure lower bound."
                    )
                gas_closure = IdealGasClosure(
                    gas_symbols,
                    self.symbols.temperature,
                    self.symbols.pressure,
                    "ideal_gas",
                )
            else:
                gas_closure = IdealGasClosure(
                    gas_symbols,
                    self.symbols.temperature,
                    self.symbols.pressure,
                    "explicit",
                    explicit_minimum=gas_total.value,
                )

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "inert_species", inert_species)
        object.__setattr__(self, "species_by_id", by_id)
        object.__setattr__(self, "state_bounds", MappingProxyType(bounds))
        object.__setattr__(self, "gas_closure", gas_closure)

    @property
    def states(self) -> CaseSymbols:
        """Compatibility name for checks awaiting their later-phase rewrite."""

        return self.symbols

    @property
    def parameter_bounds(self) -> Mapping[sp.Symbol, ParameterRange]:
        return MappingProxyType(
            {
                self.symbols.temperature: self.parameters.temperature,
                self.symbols.pressure: self.parameters.pressure,
            }
        )


__all__ = (
    "Case",
    "CheckConfig",
    "ConcentrationModel",
    "DomainConfig",
    "ParameterBox",
    "ParameterRange",
    "ReportConfig",
    "TotalMinimumConfig",
    "TotalMinimumMode",
)
