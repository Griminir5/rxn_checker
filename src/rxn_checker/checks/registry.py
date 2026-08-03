"""Explicit ordered registry of checks run by the CLI."""

from .atom_conservation import CHECK as ATOM_CONSERVATION_CHECK
from .mass_conservation import CHECK as MASS_CONSERVATION_CHECK

CHECK_REGISTRY = (
    ATOM_CONSERVATION_CHECK,
    MASS_CONSERVATION_CHECK,
)
