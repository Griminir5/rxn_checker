from pathlib import Path

import yaml
from sympy import Symbol

from .case import Case
from .reaction import Reaction
from .reactions import FAMILY_REGISTRY, REACTION_REGISTRY, ReactionBuilder
from .species import PROPERTY_REGISTRY, PropertyRegistry
from .state import StateVariables, VariableBounds


def _string_list(
    config: object,
    key: str,
    *,
    optional: bool = False,
) -> tuple[str, ...]:
    values = config.get(key, []) if isinstance(config, dict) else None
    if not isinstance(values, list) or (not optional and key not in config):
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    values = tuple(values)
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
    family_id, reaction_name = reaction_id.split(".", 1)
    if reaction.name != reaction_name:
        raise ValueError(
            f"Builder for '{reaction_id}' returned unexpected name '{reaction.name}'."
        )

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


def _number(config: dict, key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Case bound '{key}' must be numeric.")
    return float(value)


def _bounds_pair(config: dict, key: str) -> tuple[float, float]:
    values = config.get(key)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"Case bound '{key}' must contain [lower, upper].")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in values
    ):
        raise ValueError(f"Case bound '{key}' must contain numeric values.")
    return float(values[0]), float(values[1])


def _load_state_bounds(
    config: object,
    states: StateVariables,
) -> dict[Symbol, VariableBounds]:
    if not isinstance(config, dict):
        raise ValueError("Case 'bounds' must be a YAML mapping.")

    temperature = VariableBounds(*_bounds_pair(config, "temperature"))
    pressure = VariableBounds(*_bounds_pair(config, "pressure"))
    if temperature.physical_lower <= 0:
        raise ValueError("Temperature lower bound must be positive.")
    if pressure.physical_lower < 0:
        raise ValueError("Pressure lower bound must be non-negative.")

    concentrations = config.get("concentrations")
    if not isinstance(concentrations, dict):
        raise ValueError("Case concentration bounds must be a YAML mapping.")
    defaults = concentrations.get("default")
    if not isinstance(defaults, dict):
        raise ValueError("Case concentration bounds require a 'default' mapping.")
    default_upper = _number(defaults, "upper")
    default_excursion = _number(defaults, "excursion_lower")

    overrides = concentrations.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("Case concentration 'overrides' must be a YAML mapping.")
    unknown_species = set(overrides) - set(states.species_ids)
    if unknown_species:
        raise ValueError(
            "Concentration bounds reference unknown species: "
            + ", ".join(sorted(unknown_species))
            + "."
        )

    state_bounds = {
        states.temperature: temperature,
        states.pressure: pressure,
    }
    for species_id in states.species_ids:
        override = overrides.get(species_id, {})
        if not isinstance(override, dict):
            raise ValueError(
                f"Concentration bounds for '{species_id}' must be a YAML mapping."
            )
        upper = float(override.get("upper", default_upper))
        excursion_lower = float(override.get("excursion_lower", default_excursion))
        state_bounds[states.concentration(species_id)] = VariableBounds(
            physical_lower=0.0,
            physical_upper=upper,
            excursion_lower=excursion_lower,
        )
    return state_bounds


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
    inert_species = _string_list(config, "inerts", optional=True)
    reaction_selectors = _string_list(config, "reactions")
    missing_species = [
        species_id
        for species_id in species_ids
        if species_id not in property_registry.records
    ]
    if missing_species:
        raise ValueError("Unknown case species: " + ", ".join(missing_species) + ".")

    states = StateVariables(species_ids)
    reactions = _load_reactions(reaction_selectors, states)
    state_bounds = _load_state_bounds(config.get("bounds"), states)
    return Case(
        name=path.parent.name,
        states=states,
        reactions=reactions,
        state_bounds=state_bounds,
        inert_species=inert_species,
    )
