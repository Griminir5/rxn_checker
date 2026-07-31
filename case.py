from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from sympy import Symbol
import yaml

from reactions import Reaction
from species import PROPERTY_REGISTRY, PropertyRegistry


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


@dataclass(frozen=True)
class Case:
    name: str
    reaction_ids: tuple[str, ...]
    states: StateVariables
    reactions: tuple[Reaction, ...]

    def __post_init__(self) -> None:
        reaction_ids = [reaction.id for reaction in self.reactions]
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Case reaction ids must be unique.")

        available_species = set(self.states.species_ids)
        for reaction in self.reactions:
            missing = set(reaction.species_ids) - available_species
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Reaction '{reaction.id}' requires missing species: {names}."
                )
            if not reaction.rate.free_symbols <= self.states.symbols:
                raise ValueError(
                    f"Reaction '{reaction.id}' does not use this case's state symbols."
                )


def _string_list(config: object, key: str) -> tuple[str, ...]:
    if not isinstance(config, dict) or not isinstance(config.get(key), list):
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    values = tuple(config[key])
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Case '{key}' entries must be non-empty strings.")
    return values


ReactionBuilder = Callable[[StateVariables], Reaction]


def _parse_reaction_selector(selector: str) -> tuple[str, str | None]:
    parts = selector.split(".")
    if len(parts) not in (1, 2) or any(not part.isidentifier() for part in parts):
        raise ValueError(
            f"Invalid reaction selector '{selector}'; expected "
            "'family' or 'family.reaction'."
        )
    family_id = parts[0]
    reaction_name = parts[1] if len(parts) == 2 else None
    return family_id, reaction_name


def _family_builders(family_id: str) -> dict[str, ReactionBuilder]:
    module_name = f"reactions.{family_id}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Unknown reaction family '{family_id}'.") from exc
        raise

    try:
        declared_builders = module.REACTIONS
    except AttributeError as exc:
        raise AttributeError(
            f"Reaction family module '{family_id}' must define REACTIONS."
        ) from exc
    if not isinstance(declared_builders, Mapping) or not declared_builders:
        raise TypeError(
            f"Reaction family module '{family_id}' must define REACTIONS as a "
            "non-empty mapping."
        )

    builders: dict[str, ReactionBuilder] = {}
    for reaction_name, builder in declared_builders.items():
        if not isinstance(reaction_name, str) or not reaction_name.isidentifier():
            raise ValueError(
                f"Reaction family '{family_id}' has invalid reaction name "
                f"'{reaction_name}'."
            )
        if not callable(builder):
            raise TypeError(
                f"Builder for '{family_id}.{reaction_name}' must be callable."
            )
        builders[reaction_name] = builder
    return builders


def _build_reaction(
    family_id: str,
    reaction_name: str,
    builder: ReactionBuilder,
    states: StateVariables,
) -> Reaction:
    reaction_id = f"{family_id}.{reaction_name}"
    reaction = builder(states)
    if not isinstance(reaction, Reaction):
        raise TypeError(f"Builder for '{reaction_id}' did not return a Reaction.")
    if reaction.id != reaction_id:
        raise ValueError(
            f"Builder for '{reaction_id}' returned reaction '{reaction.id}'."
        )
    if reaction.family != family_id:
        raise ValueError(
            f"Reaction '{reaction_id}' declared family '{reaction.family}'."
        )
    return reaction


def _load_reactions(
    selectors: tuple[str, ...], states: StateVariables
) -> tuple[Reaction, ...]:
    reactions: list[Reaction] = []
    selected_ids: set[str] = set()
    for selector in selectors:
        family_id, requested_name = _parse_reaction_selector(selector)
        builders = _family_builders(family_id)
        if requested_name is None:
            selected_builders = builders.items()
        else:
            try:
                selected_builders = ((requested_name, builders[requested_name]),)
            except KeyError as exc:
                available = ", ".join(builders)
                raise ValueError(
                    f"Unknown reaction '{selector}'. Available reactions in "
                    f"'{family_id}': {available}."
                ) from exc

        for reaction_name, builder in selected_builders:
            reaction_id = f"{family_id}.{reaction_name}"
            if reaction_id in selected_ids:
                raise ValueError(
                    f"Reaction '{reaction_id}' was selected more than once."
                )
            selected_ids.add(reaction_id)
            reactions.append(_build_reaction(family_id, reaction_name, builder, states))
    return tuple(reactions)


def load_case(
    path: str | Path,
    *,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
) -> Case:
    """Load a case and pass its state variables into each reaction module."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    species_ids = _string_list(config, "species")
    reaction_selectors = _string_list(config, "reactions")
    missing_species = [
        species_id
        for species_id in species_ids
        if not property_registry.has_species(species_id)
    ]
    if missing_species:
        raise ValueError("Unknown case species: " + ", ".join(missing_species) + ".")

    states = StateVariables(species_ids)
    reactions = _load_reactions(reaction_selectors, states)
    return Case(
        name=path.parent.name,
        reaction_ids=tuple(reaction.id for reaction in reactions),
        states=states,
        reactions=reactions,
    )


__all__ = ("Case", "StateVariables", "load_case")
