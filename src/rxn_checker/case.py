from dataclasses import dataclass

from .reaction import Reaction
from .state import StateVariables


@dataclass(frozen=True)
class Case:
    """A validated set of states and selected reactions."""

    name: str
    states: StateVariables
    reactions: tuple[Reaction, ...]

    def __post_init__(self) -> None:
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
