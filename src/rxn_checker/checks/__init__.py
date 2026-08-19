"""Static check definitions and dependency-aware execution."""

from .definitions import CheckScope, CheckSpec, Stage
from .registry import CHECK_REGISTRY, PROFILES, plan_checks, validate_registry
from .runner import execute_plan, run_checks

__all__ = (
    "CHECK_REGISTRY",
    "PROFILES",
    "CheckScope",
    "CheckSpec",
    "Stage",
    "execute_plan",
    "plan_checks",
    "run_checks",
    "validate_registry",
)
