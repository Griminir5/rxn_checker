"""Generic execution and isolation for registered checks."""

from collections.abc import Iterable

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckExecution,
    CheckOutcome,
    CheckStatus,
)


def _normalise_outcomes(returned: object) -> tuple[CheckOutcome, ...]:
    if isinstance(returned, CheckOutcome):
        outcomes = (returned,)
    else:
        if isinstance(returned, (str, bytes)) or not isinstance(returned, Iterable):
            raise TypeError(
                "Check runners must return a CheckOutcome or an iterable of them."
            )
        outcomes = tuple(returned)

    if not outcomes:
        raise ValueError("Check runner returned no outcomes.")
    if any(not isinstance(outcome, CheckOutcome) for outcome in outcomes):
        raise TypeError("Check runner returned an invalid outcome.")
    return outcomes


def _indeterminate(error: Exception) -> tuple[CheckOutcome, ...]:
    message = str(error.args[0]) if len(error.args) == 1 else str(error)
    return (
        CheckOutcome(
            status=CheckStatus.INDETERMINATE,
            details=(f"{type(error).__name__}: {message}",),
        ),
    )


def run_checks(
    case: Case,
    checks: Iterable[CheckDefinition] | None = None,
    *,
    context: CheckContext | None = None,
) -> tuple[CheckExecution, ...]:
    """Run checks in order, isolating an unexpected failure to its check."""

    if checks is None:
        from .registry import CHECK_REGISTRY

        checks = CHECK_REGISTRY
    definitions = tuple(checks)
    check_ids = tuple(definition.id for definition in definitions)
    if len(check_ids) != len(set(check_ids)):
        raise ValueError("Registered check ids must be unique.")

    context = context or CheckContext()
    executions: list[CheckExecution] = []
    for definition in definitions:
        try:
            outcomes = _normalise_outcomes(definition.run(case, context))
        except Exception as error:
            outcomes = _indeterminate(error)
        executions.append(CheckExecution(definition, outcomes))
    return tuple(executions)
