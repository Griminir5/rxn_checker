"""Common types shared by every rxn-checker check."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import TYPE_CHECKING, Protocol

from ..species import PROPERTY_REGISTRY, PropertyRegistry

if TYPE_CHECKING:
    from ..case import Case


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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("Check value names must not be blank or padded.")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("Check values must be real numbers.")
        if self.unit is not None and (
            not isinstance(self.unit, str)
            or not self.unit
            or self.unit != self.unit.strip()
        ):
            raise ValueError("Check value units must not be blank or padded.")


@dataclass(frozen=True)
class CheckOutcome:
    """One status and/or collection of values for a case or reaction."""

    status: CheckStatus | None = None
    subject: str | None = None
    details: tuple[str, ...] = ()
    values: tuple[CheckValue, ...] = ()

    def __post_init__(self) -> None:
        details = tuple(self.details)
        values = tuple(self.values)
        if self.status is not None and not isinstance(self.status, CheckStatus):
            raise TypeError("Check outcome status must be a CheckStatus or None.")
        if self.subject is not None and (
            not isinstance(self.subject, str)
            or not self.subject
            or self.subject != self.subject.strip()
        ):
            raise ValueError("Check outcome subjects must not be blank or padded.")
        if any(not detail or not isinstance(detail, str) for detail in details):
            raise ValueError("Check outcome details must be non-empty strings.")
        if any(not isinstance(value, CheckValue) for value in values):
            raise TypeError("Check outcome values must be CheckValue instances.")
        if self.status is None and not details and not values:
            raise ValueError(
                "A status-free check outcome must provide details or values."
            )
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class CheckContext:
    """Shared dependencies supplied to every check runner."""

    property_registry: PropertyRegistry = PROPERTY_REGISTRY


CheckReturn = CheckOutcome | Iterable[CheckOutcome]


class CheckRunner(Protocol):
    def __call__(self, case: Case, context: CheckContext) -> CheckReturn:
        """Run a check against a fully loaded case."""

        ...


@dataclass(frozen=True)
class CheckDefinition:
    """Metadata and runner for one independently implemented check."""

    id: str
    name: str
    group: str
    scope: CheckScope
    run: CheckRunner = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or any(not part.isidentifier() for part in self.id.split("."))
        ):
            raise ValueError(
                "Check ids must contain identifier components separated by dots."
            )
        for label, value in (("name", self.name), ("group", self.group)):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"Check {label} must not be blank or padded.")
        if not isinstance(self.scope, CheckScope):
            raise TypeError("Check scope must be a CheckScope.")
        if not callable(self.run):
            raise TypeError("Check runner must be callable.")


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


__all__ = (
    "CheckContext",
    "CheckDefinition",
    "CheckExecution",
    "CheckOutcome",
    "CheckReturn",
    "CheckRunner",
    "CheckScope",
    "CheckStatus",
    "CheckValue",
    "aggregate_status",
)
