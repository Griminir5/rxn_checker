"""Explicit ordered registry of checks run by the CLI."""

from .atom_conservation import CHECK as ATOM_CONSERVATION_CHECK
from .equilibria import CHECK as EQUILIBRIA_CHECK
from .lipschitz_continuity import CHECK as LIPSCHITZ_CONTINUITY_CHECK
from .mass_conservation import CHECK as MASS_CONSERVATION_CHECK
from .negative_side_recovery import CHECK as NEGATIVE_SIDE_RECOVERY_CHECK
from .nonnegative_rate import CHECK as RATE_NONNEGATIVITY_CHECK
from .nonphysical_recovery import CHECK as NONPHYSICAL_RECOVERY_CHECK
from .stoichiometric_conservation import CHECK as STOICHIOMETRIC_CONSERVATION_CHECK
from .terminal_faces import CHECK as TERMINAL_FACES_CHECK
from .zero_at_depletion import CHECK as ZERO_AT_DEPLETION_CHECK

CHECK_REGISTRY = (
    ATOM_CONSERVATION_CHECK,
    MASS_CONSERVATION_CHECK,
    RATE_NONNEGATIVITY_CHECK,
    LIPSCHITZ_CONTINUITY_CHECK,
    ZERO_AT_DEPLETION_CHECK,
    NONPHYSICAL_RECOVERY_CHECK,
    NEGATIVE_SIDE_RECOVERY_CHECK,
    STOICHIOMETRIC_CONSERVATION_CHECK,
    EQUILIBRIA_CHECK,
    TERMINAL_FACES_CHECK,
)
