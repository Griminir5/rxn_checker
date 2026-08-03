from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sympy import Symbol

from .reaction import Reaction
from .state import StateVariables, VariableBounds


@dataclass(frozen=True)
class Case:
    """A validated set of states and selected reactions."""

    name: str
    states: StateVariables
    reactions: tuple[Reaction, ...]
    state_bounds: Mapping[Symbol, VariableBounds]

    def __post_init__(self) -> None:
        if not self.reactions:
            raise ValueError("Case must contain at least one reaction.")

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
        object.__setattr__(
            self,
            "state_bounds",
            MappingProxyType(state_bounds),
        )

        available_species = set(self.states.species_ids)
        for reaction in self.reactions:
            missing = set(reaction.species_ids) - available_species
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: {names}."
                )

            unknown_symbols = reaction.rate.free_symbols - self.states.symbols
            if unknown_symbols:
                names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
                raise ValueError(
                    f"Reaction '{reaction.id}' uses symbols not owned by this case: "
                    f"{names}."
                )
