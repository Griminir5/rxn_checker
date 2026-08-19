"""Static check metadata used by planning and execution."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..context import AnalysisContext
from ..results import CheckResult, Finding, Role


class Stage(StrEnum):
    CHEMISTRY = "chemistry"
    PHYSICAL = "physical"
    AUGMENTED = "augmented"
    ANALYSIS = "analysis"


class CheckScope(StrEnum):
    REACTION = "reaction"
    CASE = "case"


CheckReturn = Finding | Iterable[Finding]
CheckRunner = Callable[[AnalysisContext, Mapping[str, CheckResult]], CheckReturn]


@dataclass(frozen=True)
class CheckSpec:
    id: str
    name: str
    stage: Stage
    scope: CheckScope
    requires: tuple[str, ...]
    blocking: bool
    profiles: frozenset[str]
    run: CheckRunner
    role: Role | None = None

    def __post_init__(self) -> None:
        if not self.id.isidentifier():
            raise ValueError(f"Invalid check id '{self.id}'.")
        object.__setattr__(self, "stage", Stage(self.stage))
        object.__setattr__(self, "scope", CheckScope(self.scope))
        object.__setattr__(self, "requires", tuple(self.requires))
        if not isinstance(self.blocking, bool):
            raise TypeError("Check blocking metadata must be boolean.")
        role = self.role
        if role is None:
            role = (
                Role.BLOCKING
                if self.blocking
                else Role.ANALYSIS
                if self.stage is Stage.ANALYSIS
                else Role.ADVISORY
            )
        role = Role(role)
        if self.blocking != (role is Role.BLOCKING):
            raise ValueError("Check role and blocking metadata disagree.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "profiles", frozenset(self.profiles))


__all__ = ("CheckReturn", "CheckRunner", "CheckScope", "CheckSpec", "Stage")
