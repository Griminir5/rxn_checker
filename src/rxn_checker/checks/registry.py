"""Explicit ordered registry of checks run by the CLI."""

from .atom_conservation import CHECK as ATOM_CONSERVATION_CHECK
from .mass_conservation import CHECK as MASS_CONSERVATION_CHECK
from .network_positivity import CHECK as NETWORK_POSITIVITY_CHECK
from .nonnegative_rate import CHECK as RATE_NONNEGATIVITY_CHECK
from .zero_at_depletion import CHECK as ZERO_AT_DEPLETION_CHECK

CHECK_REGISTRY = (
    ATOM_CONSERVATION_CHECK,
    MASS_CONSERVATION_CHECK,
    RATE_NONNEGATIVITY_CHECK,
    ZERO_AT_DEPLETION_CHECK,
    NETWORK_POSITIVITY_CHECK,
)
