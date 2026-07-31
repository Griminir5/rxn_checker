from __future__ import annotations

from pathlib import Path

import yaml

from .case import Case
from .reaction import Reaction
from .reactions import FAMILY_REGISTRY, REACTION_REGISTRY, ReactionBuilder
from .species import PROPERTY_REGISTRY, PropertyRegistry
from .state import StateVariables


def _string_list(config: object, key: str) -> tuple[str, ...]:
    if not isinstance(config, dict) or not isinstance(config.get(key), list):
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    values = tuple(config[key])
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Case '{key}' entries must be non-empty strings.")
    return values


def _selected_builders(
    selectors: tuple[str, ...],
) -> tuple[tuple[str, ReactionBuilder], ...]:
    selected: list[tuple[str, ReactionBuilder]] = []
    selected_ids: set[str] = set()

    for selector in selectors:
        parts = selector.split(".")
        if len(parts) not in (1, 2) or any(not part.isidentifier() for part in parts):
            raise ValueError(
                f"Invalid reaction selector '{selector}'; expected "
                "'family' or 'family.reaction'."
            )

        if len(parts) == 1:
            try:
                reaction_ids = FAMILY_REGISTRY[selector]
            except KeyError as exc:
                raise ValueError(f"Unknown reaction family '{selector}'.") from exc
        else:
            reaction_ids = (selector,)
            if selector not in REACTION_REGISTRY:
                family_id = parts[0]
                available = ", ".join(FAMILY_REGISTRY.get(family_id, ()))
                message = f"Unknown reaction '{selector}'."
                if available:
                    message += f" Available reactions: {available}."
                raise ValueError(message)

        for reaction_id in reaction_ids:
            if reaction_id in selected_ids:
                raise ValueError(
                    f"Reaction '{reaction_id}' was selected more than once."
                )
            selected_ids.add(reaction_id)
            selected.append((reaction_id, REACTION_REGISTRY[reaction_id]))

    return tuple(selected)


def _build_reaction(
    reaction_id: str,
    builder: ReactionBuilder,
    states: StateVariables,
) -> Reaction:
    reaction = builder(states)
    if not isinstance(reaction, Reaction):
        raise TypeError(f"Builder for '{reaction_id}' did not return a Reaction.")
    if reaction.id != reaction_id:
        raise ValueError(
            f"Builder for '{reaction_id}' returned reaction '{reaction.id}'."
        )

    family_id = reaction_id.split(".", 1)[0]
    if reaction.family != family_id:
        raise ValueError(
            f"Reaction '{reaction_id}' declared family '{reaction.family}'."
        )
    return reaction


def _load_reactions(
    selectors: tuple[str, ...], states: StateVariables
) -> tuple[Reaction, ...]:
    return tuple(
        _build_reaction(reaction_id, builder, states)
        for reaction_id, builder in _selected_builders(selectors)
    )


def load_case(
    path: str | Path,
    *,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
) -> Case:
    """Load a case and build only the reactions selected by its YAML."""

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


__all__ = ("load_case",)
