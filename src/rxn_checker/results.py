"""Structured, renderer-independent results."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class Role(StrEnum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"
    ANALYSIS = "analysis"


@dataclass(frozen=True)
class Evidence:
    """Small structured payload retained by text and JSON reports."""

    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


@dataclass(frozen=True)
class Finding:
    subject: str
    verdict: Verdict
    summary: str
    evidence: Evidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", Verdict(self.verdict))


_PRIORITY = {
    Verdict.PASS: 0,
    Verdict.SKIPPED: 1,
    Verdict.UNKNOWN: 2,
    Verdict.FAIL: 3,
    Verdict.ERROR: 4,
}


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    role: Role
    findings: tuple[Finding, ...]
    duration_seconds: float

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        if not findings:
            raise ValueError("A check result must contain at least one finding.")
        object.__setattr__(self, "role", Role(self.role))
        object.__setattr__(self, "findings", findings)

    @property
    def verdict(self) -> Verdict:
        return max(
            (finding.verdict for finding in self.findings),
            key=_PRIORITY.__getitem__,
        )


@dataclass(frozen=True)
class RunResult:
    case_name: str
    selected_checks: tuple[str, ...]
    results: Mapping[str, CheckResult]
    overall: Verdict

    def __post_init__(self) -> None:
        selected = tuple(self.selected_checks)
        results = dict(self.results)
        if tuple(results) != selected:
            raise ValueError("Run results must follow the selected check order.")
        object.__setattr__(self, "selected_checks", selected)
        object.__setattr__(self, "results", MappingProxyType(results))
        object.__setattr__(self, "overall", Verdict(self.overall))

    @property
    def passed(self) -> bool:
        return self.overall is Verdict.PASS


def overall_verdict(results: Mapping[str, CheckResult]) -> Verdict:
    if any(result.verdict is Verdict.ERROR for result in results.values()):
        return Verdict.ERROR
    blocking = tuple(
        result.verdict
        for result in results.values()
        if result.role is Role.BLOCKING
    )
    return max(blocking, key=_PRIORITY.__getitem__, default=Verdict.PASS)


__all__ = (
    "CheckResult",
    "Evidence",
    "Finding",
    "Role",
    "RunResult",
    "Verdict",
    "overall_verdict",
)
