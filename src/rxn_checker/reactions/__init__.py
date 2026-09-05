"""Explicit built-in reaction-family locations."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ..model import CaseSymbols, Reaction

FamilyBuilder = Callable[[CaseSymbols], Mapping[str, Reaction]]

# Module names, rather than imported modules, keep unselected families unloaded.
BUILTIN_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "aye_plus_bee_to_cee": "rxn_checker.reactions.aye_plus_bee_to_cee",
        "aye_to_bee": "rxn_checker.reactions.aye_to_bee",
        "medrano": "rxn_checker.reactions.medrano",
        "xu_froment": "rxn_checker.reactions.xu_froment",
    }
)

__all__ = ("BUILTIN_FAMILIES", "FamilyBuilder")
