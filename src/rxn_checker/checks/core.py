"""Built-in check graph, selection, and execution."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from ..case import Case
from ..context import AnalysisContext
from ..results import CheckResult, Finding, Role, RunResult, Verdict, overall_verdict
from .analyses import run_conserved_quantities, run_steady_state_equations, run_structural_faces
from .atom_conservation import run as atom_conservation
from .definedness import run_augmented as augmented_definedness
from .definedness import run_physical as physical_definedness
from .differential_profile import run as run_differential_solver_profile
from .evaluation_profile import run as run_evaluation_profile
from .invariance import run_boundary_inward, run_forward_invariance
from .lipschitz import run_augmented as augmented_lipschitz
from .lipschitz import run_physical as physical_lipschitz
from .mass_conservation import run as mass_conservation
from .negative_side import run as negative_side
from .nonnegative_rate import run as rate_nonnegativity
from .zero_at_depletion import run as zero_at_depletion


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
    profiles: frozenset[str]
    run: CheckRunner
    role: Role = Role.BLOCKING
    accepts_partial_dependencies: bool = False


_TARGETS = {
    "basic": ("atom_conservation", "mass_conservation"),
    "physical": ("forward_invariance",),
    "robust": ("forward_invariance", "negative_side_nonrepulsion"),
    "analysis": (
        "forward_invariance",
        "conserved_quantities",
        "structural_faces",
        "steady_state_equations",
        "evaluation_profile",
        "differential_solver_profile",
    ),
    "all": (
        "forward_invariance",
        "negative_side_nonrepulsion",
        "conserved_quantities",
        "structural_faces",
        "steady_state_equations",
        "evaluation_profile",
        "differential_solver_profile",
    ),
}
PROFILES = frozenset(_TARGETS)


def _spec(id, name, stage, scope, requires, run, *, partial=False):
    return CheckSpec(
        id,
        name,
        stage,
        scope,
        requires,
        frozenset(name for name, ids in _TARGETS.items() if id in ids),
        run,
        Role.ANALYSIS if stage is Stage.ANALYSIS else Role.BLOCKING,
        partial,
    )


R, C = CheckScope.REACTION, CheckScope.CASE
CHEM, PHYS, AUG, ANALYSIS = Stage
# fmt: off
CHECK_REGISTRY = (
    _spec("atom_conservation", "Atom conservation", CHEM, R,
          (), atom_conservation),
    _spec("mass_conservation", "Mass conservation", CHEM, R,
          (), mass_conservation),
    _spec("physical_rate_definedness", "Rate definedness", PHYS, R,
          ("atom_conservation", "mass_conservation"), physical_definedness),
    _spec("rate_nonnegativity", "Rate non-negativity", PHYS, R,
          ("physical_rate_definedness",), rate_nonnegativity),
    _spec("physical_lipschitz", "Lipschitz continuity", PHYS, R,
          ("physical_rate_definedness",), physical_lipschitz),
    _spec("zero_at_depletion", "Zero rate at depletion", PHYS, R,
          ("physical_rate_definedness",), zero_at_depletion),
    _spec("physical_boundary_inward", "Physical boundary inward", PHYS, C,
          ("rate_nonnegativity", "zero_at_depletion"), run_boundary_inward),
    _spec("forward_invariance", "Forward invariance", PHYS, C,
          ("physical_boundary_inward", "physical_lipschitz"), run_forward_invariance),
    _spec("augmented_rate_definedness", "Augmented rate definedness", AUG, R,
          ("atom_conservation", "mass_conservation"), augmented_definedness),
    _spec("augmented_lipschitz", "Augmented Lipschitz continuity", AUG, R,
          ("augmented_rate_definedness",), augmented_lipschitz),
    _spec("negative_side_nonrepulsion", "Negative-side non-repulsion", AUG, C,
          ("augmented_lipschitz",), negative_side, partial=True),
    _spec("conserved_quantities", "Conserved quantities", ANALYSIS, C,
          ("atom_conservation", "mass_conservation"), run_conserved_quantities),
    _spec("structural_faces", "Structural faces", ANALYSIS, C,
          ("atom_conservation", "mass_conservation"), run_structural_faces),
    _spec("steady_state_equations", "Steady-state equations", ANALYSIS, C,
          ("atom_conservation", "mass_conservation"), run_steady_state_equations),
    _spec("evaluation_profile", "Evaluation profile", ANALYSIS, C,
          (), run_evaluation_profile),
    _spec("differential_solver_profile", "Differential solver profile", ANALYSIS, C,
          (), run_differential_solver_profile),
)
# fmt: on


def validate_registry(registry=CHECK_REGISTRY) -> tuple[CheckSpec, ...]:
    """Validate and topologically order a registry, preserving tie order."""
    specs = tuple(registry)
    by_id = {spec.id: spec for spec in specs}
    if len(by_id) != len(specs):
        raise ValueError("Registered check ids must be unique.")
    stage_order = {stage: index for index, stage in enumerate(Stage)}
    for spec in specs:
        if len(spec.requires) != len(set(spec.requires)):
            raise ValueError(f"Check '{spec.id}' repeats a dependency.")
        for required in spec.requires:
            if required in by_id and stage_order[by_id[required].stage] > stage_order[spec.stage]:
                raise ValueError(f"Check '{spec.id}' depends on a later stage '{required}'.")
    ordered, visiting, visited = [], set(), set()

    def visit(check_id):
        if check_id in visiting:
            raise ValueError(f"Check dependency cycle includes '{check_id}'.")
        if check_id in visited:
            return
        try:
            spec = by_id[check_id]
        except KeyError as error:
            raise ValueError(f"Check has unknown dependencies: {check_id}.") from error
        visiting.add(check_id)
        for required in spec.requires:
            visit(required)
        visiting.remove(check_id)
        visited.add(check_id)
        ordered.append(spec)

    for spec in sorted(specs, key=lambda spec: stage_order[spec.stage]):
        visit(spec.id)
    return tuple(ordered)


def _closure(ids, by_id):
    selected = set()

    def add(check_id):
        if check_id not in selected:
            selected.add(check_id)
            for required in by_id[check_id].requires:
                add(required)

    for check_id in ids:
        add(check_id)
    return selected


def plan_checks(
    *, profile="physical", include=(), exclude=(), only=None, registry=CHECK_REGISTRY
) -> tuple[CheckSpec, ...]:
    ordered = validate_registry(registry)
    by_id = {spec.id: spec for spec in ordered}
    explicit = tuple(include) if only is None else tuple(only)
    if only is None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown check profile '{profile}'.")
        requested = set(explicit) | {spec.id for spec in ordered if profile in spec.profiles}
    else:
        requested = set(explicit)
    excluded = set(exclude)
    unknown = (requested | excluded) - set(by_id)
    if unknown:
        raise ValueError("Unknown checks: " + ", ".join(sorted(unknown)) + ".")
    conflict = _closure(explicit, by_id) & excluded
    if conflict:
        raise ValueError(
            "Explicitly selected checks require excluded checks: "
            + ", ".join(sorted(conflict))
            + "."
        )
    requested = _closure(requested, by_id)
    selected = set()
    for spec in ordered:
        if spec.id in requested and spec.id not in excluded and set(spec.requires) <= selected:
            selected.add(spec.id)
    return tuple(spec for spec in ordered if spec.id in selected)


def _skip(spec, case, summary):
    return CheckResult(spec.id, spec.role, (Finding(case.name, Verdict.SKIPPED, summary),), 0.0)


def execute_plan(case: Case, plan, *, context=None, fail_fast="stage", debug=False) -> RunResult:
    """Run a dependency-ordered plan; analyses still run after blocking failures."""
    if fail_fast not in {"stage", "none"}:
        raise ValueError("fail_fast must be 'stage' or 'none'.")
    specs = validate_registry(plan)
    context = context or AnalysisContext(case)
    results, stopped_after, current, stage_failed = {}, None, None, False
    reaction_ids = {reaction.id for reaction in case.reactions}
    by_id = {spec.id: spec for spec in specs}
    for spec in specs:
        if spec.stage is not current:
            if current is not None and stage_failed and fail_fast == "stage":
                stopped_after = current
            current, stage_failed = spec.stage, False
        if stopped_after is not None and spec.stage is not Stage.ANALYSIS:
            result = _skip(spec, case, f"The {stopped_after.value} stage did not pass.")
        else:
            failed = tuple(r for r in spec.requires if results[r].verdict is not Verdict.PASS)
            partial = (spec.scope is R or spec.accepts_partial_dependencies) and all(
                by_id[r].scope is R
                and by_id[r].stage is spec.stage
                and reaction_ids <= {finding.subject for finding in results[r].findings}
                for r in failed
            )
            if failed and not partial:
                reason = ", ".join(f"{r}={results[r].verdict.value}" for r in failed)
                result = _skip(spec, case, f"Requires passing prerequisites: {reason}.")
            else:
                started = perf_counter()
                try:
                    value = spec.run(context, {r: results[r] for r in spec.requires})
                    findings = (value,) if isinstance(value, Finding) else tuple(value)
                    if not findings or any(not isinstance(item, Finding) for item in findings):
                        raise TypeError("Check runners must return one or more findings.")
                except Exception as error:
                    if debug:
                        raise
                    findings = (
                        Finding(case.name, Verdict.ERROR, f"{type(error).__name__}: {error}"),
                    )
                result = CheckResult(spec.id, spec.role, findings, perf_counter() - started)
        results[spec.id] = result
        stage_failed |= spec.role is Role.BLOCKING and result.verdict in {
            Verdict.FAIL,
            Verdict.ERROR,
        }
    ids = tuple(spec.id for spec in specs)
    return RunResult(case.name, ids, results, overall_verdict(results))


def run_checks(
    case: Case,
    checks=None,
    *,
    context=None,
    profile=None,
    include=None,
    exclude=None,
    only=None,
    fail_fast=None,
    debug=False,
) -> RunResult:
    config = case.check_config
    registry = tuple(checks) if checks is not None else CHECK_REGISTRY
    no_selection = profile is None and include is None and exclude is None and only is None
    plan = (
        validate_registry(registry)
        if checks is not None and no_selection
        else plan_checks(
            profile=profile or config.get("profile", "physical"),
            include=include if include is not None else config.get("include", ()),
            exclude=exclude if exclude is not None else config.get("exclude", ()),
            only=only,
            registry=registry,
        )
    )
    return execute_plan(
        case,
        plan,
        context=context,
        fail_fast=fail_fast or config.get("fail_fast", "stage"),
        debug=debug,
    )
