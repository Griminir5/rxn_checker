"""Dependency-aware execution of a validated check plan."""

from collections.abc import Iterable, Mapping
from time import perf_counter

from ..case import Case
from ..context import AnalysisContext
from ..results import CheckResult, Finding, RunResult, Verdict, overall_verdict
from .definitions import CheckScope, CheckSpec, Stage
from .registry import CHECK_REGISTRY, plan_checks, validate_registry


def _findings(value: object) -> tuple[Finding, ...]:
    if isinstance(value, Finding):
        findings = (value,)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise TypeError("Check runners must return findings.")
        findings = tuple(value)
    if not findings or any(not isinstance(item, Finding) for item in findings):
        raise TypeError("Check runners must return one or more Finding objects.")
    return findings


def _skip(spec: CheckSpec, case: Case, summary: str) -> CheckResult:
    return CheckResult(
        spec.id,
        spec.role,
        (Finding(case.name, Verdict.SKIPPED, summary),),
        0.0,
    )


def execute_plan(
    case: Case,
    plan: Iterable[CheckSpec],
    *,
    context: AnalysisContext | None = None,
    fail_fast: str = "stage",
    debug: bool = False,
) -> RunResult:
    """Execute each selected check at most once and retain structured results."""

    if fail_fast not in {"stage", "none"}:
        raise ValueError("fail_fast must be 'stage' or 'none'.")
    specs = validate_registry(plan)
    by_id = {spec.id: spec for spec in specs}
    reaction_ids = frozenset(reaction.id for reaction in case.reactions)
    ids = tuple(spec.id for spec in specs)
    if len(ids) != len(set(ids)):
        raise ValueError("A check plan cannot contain duplicate ids.")
    selected = set(ids)
    for spec in specs:
        missing = set(spec.requires) - selected
        if missing:
            raise ValueError(
                f"Selected check '{spec.id}' is missing prerequisites: "
                + ", ".join(sorted(missing))
                + "."
            )

    context = context or AnalysisContext(case)
    if context.case is not case:
        raise ValueError("Analysis context belongs to a different case.")
    results: dict[str, CheckResult] = {}
    current_stage: Stage | None = None
    stage_failed = False
    stopped_after: Stage | None = None

    for spec in specs:
        if spec.stage is not current_stage:
            if current_stage is not None and stage_failed and fail_fast == "stage":
                stopped_after = current_stage
            current_stage = spec.stage
            stage_failed = False

        if stopped_after is not None and spec.stage is not Stage.ANALYSIS:
            result = _skip(
                spec,
                case,
                f"The {stopped_after.value} stage did not pass.",
            )
        else:
            failed_dependencies = tuple(
                (required, results[required].verdict)
                for required in spec.requires
                if results[required].verdict is not Verdict.PASS
            )
            accepts_partial = (
                (spec.scope is CheckScope.REACTION)
                or spec.accepts_partial_dependencies
            ) and all(
                by_id[required].scope is CheckScope.REACTION
                and by_id[required].stage is spec.stage
                and reaction_ids.issubset(
                    finding.subject for finding in results[required].findings
                )
                for required, _verdict in failed_dependencies
            )
            if failed_dependencies and not accepts_partial:
                reason = ", ".join(
                    f"{check_id}={verdict.value}"
                    for check_id, verdict in failed_dependencies
                )
                result = _skip(spec, case, f"Requires passing prerequisites: {reason}.")
            else:
                started = perf_counter()
                try:
                    dependencies: Mapping[str, CheckResult] = {
                        required: results[required] for required in spec.requires
                    }
                    findings = _findings(spec.run(context, dependencies))
                except Exception as error:
                    if debug:
                        raise
                    message = str(error.args[0]) if len(error.args) == 1 else str(error)
                    findings = (
                        Finding(
                            case.name,
                            Verdict.ERROR,
                            f"{type(error).__name__}: {message}",
                        ),
                    )
                result = CheckResult(
                    spec.id,
                    spec.role,
                    findings,
                    perf_counter() - started,
                )
        results[spec.id] = result
        if spec.blocking and result.verdict in {Verdict.FAIL, Verdict.ERROR}:
            stage_failed = True

    return RunResult(case.name, ids, results, overall_verdict(results))


def run_checks(
    case: Case,
    checks: Iterable[CheckSpec] | None = None,
    *,
    context: AnalysisContext | None = None,
    profile: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    only: Iterable[str] | None = None,
    fail_fast: str | None = None,
    debug: bool = False,
) -> RunResult:
    """Plan and run configured checks, or execute every supplied custom check."""

    registry = tuple(checks) if checks is not None else CHECK_REGISTRY
    config = case.check_config
    configured_profile = str(config.get("profile", "physical"))
    configured_include = tuple(config.get("include", ()))
    configured_exclude = tuple(config.get("exclude", ()))
    configured_fail_fast = str(config.get("fail_fast", "stage"))

    no_selection = profile is None and include is None and exclude is None and only is None
    if checks is not None and no_selection:
        plan = validate_registry(registry)
    else:
        plan = plan_checks(
            profile=profile if profile is not None else configured_profile,
            include=include if include is not None else configured_include,
            exclude=exclude if exclude is not None else configured_exclude,
            only=only,
            registry=registry,
        )
    return execute_plan(
        case,
        plan,
        context=context,
        fail_fast=fail_fast or configured_fail_fast,
        debug=debug,
    )


__all__ = ("execute_plan", "run_checks")
