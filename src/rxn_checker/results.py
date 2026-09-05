"""Structured, renderer-independent results."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


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
    kind: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class Finding:
    subject: str
    verdict: Verdict
    summary: str
    evidence: Evidence | None = None


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

    @property
    def verdict(self) -> Verdict:
        return max((finding.verdict for finding in self.findings), key=_PRIORITY.__getitem__)


@dataclass(frozen=True)
class RunResult:
    case_name: str
    selected_checks: tuple[str, ...]
    results: Mapping[str, CheckResult]
    overall: Verdict

    @property
    def passed(self) -> bool:
        return self.overall is Verdict.PASS


def overall_verdict(results: Mapping[str, CheckResult]) -> Verdict:
    if any(result.verdict is Verdict.ERROR for result in results.values()):
        return Verdict.ERROR
    blocking = (result.verdict for result in results.values() if result.role is Role.BLOCKING)
    return max(blocking, key=_PRIORITY.__getitem__, default=Verdict.PASS)
