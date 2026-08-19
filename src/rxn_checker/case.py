from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from sympy import Symbol

from .reaction import Reaction
from .state import IdealGasClosure, StateVariables, VariableBounds


@dataclass(frozen=True)
class Case:
    """A validated set of states and selected reactions."""

    name: str
    states: StateVariables
    reactions: tuple[Reaction, ...]
    state_bounds: Mapping[Symbol, VariableBounds]
    inert_species: tuple[str, ...] = ()
    gas_closure: IdealGasClosure | None = None

    def __post_init__(self) -> None:
        if not self.reactions:
            raise ValueError("Case must contain at least one reaction.")

        inert_species = tuple(self.inert_species)
        if any(not isinstance(species_id, str) for species_id in inert_species):
            raise ValueError("Case inert species must be strings.")
        inert_set = set(inert_species)
        if len(inert_species) != len(inert_set):
            raise ValueError("Case inert species must not contain duplicates.")
        unknown_inerts = inert_set - set(self.states.species_ids)
        if unknown_inerts:
            names = ", ".join(sorted(unknown_inerts))
            raise ValueError(f"Case references unknown inert species: {names}.")

        state_bounds = dict(self.state_bounds)
        bound_symbols = set(state_bounds)
        missing_bounds = self.states.symbols - bound_symbols
        unknown_bounds = bound_symbols - self.states.symbols
        if missing_bounds:
            names = ", ".join(sorted(str(symbol) for symbol in missing_bounds))
            raise ValueError(f"Case is missing bounds for state variables: {names}.")
        if unknown_bounds:
            names = ", ".join(sorted(str(symbol) for symbol in unknown_bounds))
            raise ValueError(f"Case has bounds for unknown state variables: {names}.")

        if self.gas_closure is not None:
            closure = self.gas_closure
            unknown_gases = set(closure.gas_concentrations) - set(
                self.states.concentrations.values()
            )
            if unknown_gases:
                names = ", ".join(sorted(map(str, unknown_gases)))
                raise ValueError(
                    "Ideal-gas closure references unknown concentrations: "
                    f"{names}."
                )
            if closure.temperature != self.states.temperature:
                raise ValueError("Ideal-gas closure uses the wrong temperature symbol.")
            if closure.pressure != self.states.pressure:
                raise ValueError("Ideal-gas closure uses the wrong pressure symbol.")
            if state_bounds[closure.temperature].physical_lower <= 0:
                raise ValueError(
                    "Ideal-gas closure requires a positive temperature lower bound."
                )
            if state_bounds[closure.pressure].physical_lower <= 0:
                raise ValueError(
                    "Ideal-gas closure requires a positive pressure lower bound."
                )
        available_species = set(self.states.species_ids)
        for reaction in self.reactions:
            missing = set(reaction.species_ids) - available_species
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: {names}."
                )

            participating_inerts = set(reaction.species_ids) & inert_set
            if participating_inerts:
                names = ", ".join(sorted(participating_inerts))
                raise ValueError(
                    f"Inert species participate in reaction '{reaction.id}': "
                    f"{names}."
                )

            unknown_symbols = reaction.rate.free_symbols - self.states.symbols
            if unknown_symbols:
                names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
                raise ValueError(
                    f"Reaction '{reaction.id}' uses symbols not owned by this case: "
                    f"{names}."
                )

        for species_id in inert_species:
            symbol = self.states.concentration(species_id)
            state_bounds[symbol] = replace(state_bounds[symbol], strict_lower=True)
        object.__setattr__(self, "inert_species", inert_species)
        object.__setattr__(self, "state_bounds", MappingProxyType(state_bounds))
