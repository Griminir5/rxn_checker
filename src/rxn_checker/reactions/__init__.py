"""Automatically discovered reaction-family builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pkgutil import iter_modules
from types import MappingProxyType

from ..reaction import Reaction
from ..state import StateVariables

ReactionBuilder = Callable[[StateVariables], Reaction]


def _declared_builders(
    family_id: str,
    module_name: str,
) -> dict[str, ReactionBuilder]:
    module = import_module(module_name)
    try:
        declared = module.REACTIONS
    except AttributeError as exc:
        raise AttributeError(
            f"Reaction family module '{family_id}' must define REACTIONS."
        ) from exc
    if not isinstance(declared, Mapping) or not declared:
        raise TypeError(
            f"Reaction family module '{family_id}' must define REACTIONS as a "
            "non-empty mapping."
        )

    builders: dict[str, ReactionBuilder] = {}
    for reaction_name, builder in declared.items():
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


def _discover_reactions() -> tuple[
    Mapping[str, ReactionBuilder],
    Mapping[str, tuple[str, ...]],
]:
    reaction_builders: dict[str, ReactionBuilder] = {}
    family_reactions: dict[str, tuple[str, ...]] = {}

    family_modules = sorted(
        (
            module
            for module in iter_modules(__path__, f"{__name__}.")
            if not module.ispkg and not module.name.rsplit(".", 1)[-1].startswith("_")
        ),
        key=lambda module: module.name,
    )
    for family_module in family_modules:
        family_id = family_module.name.rsplit(".", 1)[-1]
        if not family_id.isidentifier():
            raise ValueError(f"Invalid reaction family module name '{family_id}'.")

        builders = _declared_builders(family_id, family_module.name)
        reaction_ids = tuple(
            f"{family_id}.{reaction_name}" for reaction_name in builders
        )
        family_reactions[family_id] = reaction_ids
        reaction_builders.update(zip(reaction_ids, builders.values(), strict=True))

    return MappingProxyType(reaction_builders), MappingProxyType(family_reactions)


REACTION_REGISTRY, FAMILY_REGISTRY = _discover_reactions()


__all__ = (
    "FAMILY_REGISTRY",
    "REACTION_REGISTRY",
    "Reaction",
    "ReactionBuilder",
)
