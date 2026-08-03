"""Generic plain-text reporting for registered checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from .case import Case
from .checks import (
    BASIC_CHECKS,
    CheckContext,
    CheckDefinition,
    CheckExecution,
    CheckOutcome,
    CheckStatus,
    aggregate_status,
    run_checks,
)
from .species import PROPERTY_REGISTRY, PropertyRegistry


@dataclass(frozen=True)
class CheckReport:
    """Rendered report plus structured executions and summary data."""

    text: str
    overall_status: CheckStatus | None
    status_counts: Mapping[CheckStatus, int]
    value_count: int
    executions: tuple[CheckExecution, ...]

    @property
    def passed(self) -> bool:
        """Whether the report is suitable for a successful CLI exit."""

        return self.overall_status is None or self.overall_status.successful


def _number(value: Real) -> str:
    if isinstance(value, int):
        return str(value)
    try:
        return format(value, ".12g")
    except TypeError:
        return format(float(value), ".12g")


def _render_outcome(outcome: CheckOutcome) -> list[str]:
    subject = outcome.subject or "Case"
    if outcome.status is None:
        lines = [f"    {subject}"]
    else:
        lines = [f"    {subject}: {outcome.status.value}"]
    lines.extend(f"      {detail}" for detail in outcome.details)
    for value in outcome.values:
        rendered = f"      {value.name}: {_number(value.value)}"
        if value.unit is not None:
            rendered += f" {value.unit}"
        lines.append(rendered)
    return lines


def _status_counts(
    executions: Iterable[CheckExecution],
) -> Mapping[CheckStatus, int]:
    counts = {status: 0 for status in CheckStatus}
    for execution in executions:
        for outcome in execution.outcomes:
            if outcome.status is not None:
                counts[outcome.status] += 1
    return MappingProxyType(counts)


def build_check_report(
    case: Case,
    *,
    source: str | Path | None = None,
    checks: Iterable[CheckDefinition] | None = None,
    context: CheckContext | None = None,
) -> CheckReport:
    """Execute any collection of check definitions and render their outcomes."""

    executions = run_checks(case, checks, context=context)
    outcomes = tuple(
        outcome for execution in executions for outcome in execution.outcomes
    )
    overall_status = aggregate_status(outcomes)
    status_counts = _status_counts(executions)
    value_count = sum(len(outcome.values) for outcome in outcomes)

    lines = ["rxn-checker report", f"Case: {case.name}"]
    if source is not None:
        lines.append(f"Source: {Path(source)}")
    lines.extend((f"Reactions: {len(case.reactions)}", "Case loading: PASS", ""))

    current_group: str | None = None
    for execution in executions:
        definition = execution.definition
        if definition.group != current_group:
            if current_group is not None:
                lines.append("")
            lines.append(definition.group)
            current_group = definition.group
        lines.append(f"  {definition.name} [{definition.id}; {definition.scope.value}]")
        for outcome in execution.outcomes:
            lines.extend(_render_outcome(outcome))

    if executions:
        lines.append("")
    overall_label = overall_status.value if overall_status is not None else "NO_STATUS"
    rendered_counts = ", ".join(
        f"{status.value}={status_counts[status]}" for status in CheckStatus
    )
    lines.extend(
        (
            "Summary",
            f"  Overall: {overall_label}",
            f"  Statuses: {rendered_counts}",
            f"  Numerical values: {value_count}",
        )
    )
    return CheckReport(
        text="\n".join(lines) + "\n",
        overall_status=overall_status,
        status_counts=status_counts,
        value_count=value_count,
        executions=executions,
    )


def build_basic_check_report(
    case: Case,
    *,
    source: str | Path | None = None,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
) -> CheckReport:
    """Compatibility wrapper for the original basic-report API."""

    return build_check_report(
        case,
        source=source,
        checks=BASIC_CHECKS,
        context=CheckContext(property_registry=property_registry),
    )


BasicCheckReport = CheckReport


__all__ = (
    "BasicCheckReport",
    "CheckReport",
    "build_basic_check_report",
    "build_check_report",
)
