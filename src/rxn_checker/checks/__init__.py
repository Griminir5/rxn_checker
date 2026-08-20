"""Static check definitions and dependency-aware execution."""
from .core import (CHECK_REGISTRY, PROFILES, CheckScope, CheckSpec, Stage,
                   execute_plan, plan_checks, run_checks, validate_registry)

__all__ = ("CHECK_REGISTRY", "PROFILES", "CheckScope", "CheckSpec", "Stage",
           "execute_plan", "plan_checks", "run_checks", "validate_registry")
