"""Static check definitions and dependency-aware execution."""

from .core import CHECK_REGISTRY as CHECK_REGISTRY
from .core import PROFILES as PROFILES
from .core import CheckScope as CheckScope
from .core import CheckSpec as CheckSpec
from .core import Stage as Stage
from .core import execute_plan as execute_plan
from .core import plan_checks as plan_checks
from .core import run_checks as run_checks
from .core import validate_registry as validate_registry
