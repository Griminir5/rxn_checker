"""Common types shared by every rxn-checker check."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real

from ..case import Case
from ..species import PROPERTY_REGISTRY, PropertyRegistry


class CheckStatus(StrEnum):
    """Qualitative conclusion produced by a check outcome."""

    PASS = "PASS"
    SAMPLED_PASS = "SAMPLED_PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"
    UNAVAILABLE = "UNAVAILABLE"

    @property
    def successful(self) -> bool:
        return self in (CheckStatus.PASS, CheckStatus.SAMPLED_PASS)


class CheckScope(StrEnum):
    """The level at which a check interprets a case."""

    REACTION = "reaction"
    CASE = "case"


@dataclass(frozen=True)
class CheckValue:
    """One named numerical value returned by a check."""

    name: str
    value: Real
    unit: str | None = None


@dataclass(frozen=True)
class CheckOutcome:
    """One status and/or collection of values for a case or reaction."""

    status: CheckStatus | None = None
    subject: str | None = None
    details: tuple[str, ...] = ()
    values: tuple[CheckValue, ...] = ()


@dataclass(frozen=True)
class CheckContext:
    """Shared dependencies supplied to every check runner."""

    property_registry: PropertyRegistry = PROPERTY_REGISTRY
    _analysis: dict[tuple[int, str], tuple[Case, object]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def cached(
        self,
        case: Case,
        key: str,
        factory: Callable[[], object],
    ) -> object:
        """Return one case-owned analysis product shared by registered checks."""

        cache_key = id(case), key
        cached = self._analysis.get(cache_key)
        if cached is not None and cached[0] is case:
            return cached[1]
        value = factory()
        self._analysis[cache_key] = case, value
        return value


CheckReturn = CheckOutcome | Iterable[CheckOutcome]
CheckRunner = Callable[[Case, CheckContext], CheckReturn]


@dataclass(frozen=True)
class CheckDefinition:
    """Metadata and runner for one independently implemented check."""

    id: str
    name: str
    group: str
    scope: CheckScope
    run: CheckRunner


@dataclass(frozen=True)
class CheckExecution:
    """Normalized outcomes produced while executing one check definition."""

    definition: CheckDefinition
    outcomes: tuple[CheckOutcome, ...]


_STATUS_PRIORITY = {
    CheckStatus.PASS: 0,
    CheckStatus.SAMPLED_PASS: 1,
    CheckStatus.UNAVAILABLE: 2,
    CheckStatus.INDETERMINATE: 3,
    CheckStatus.FAIL: 4,
}


def aggregate_status(outcomes: Iterable[CheckOutcome]) -> CheckStatus | None:
    """Return the most consequential status, ignoring numerical-only results."""

    statuses = (outcome.status for outcome in outcomes if outcome.status is not None)
    return max(statuses, key=lambda status: _STATUS_PRIORITY[status], default=None)
