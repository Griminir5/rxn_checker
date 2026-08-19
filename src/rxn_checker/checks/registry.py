"""Validated static check DAG and deterministic selection."""

from collections.abc import Iterable, Mapping

from ..context import AnalysisContext
from ..results import CheckResult, Finding, Verdict
from .atom_conservation import run as run_atom_conservation
from .definedness import run_augmented as run_augmented_definedness
from .definedness import run_physical as run_physical_definedness
from .definitions import CheckScope, CheckSpec, Stage
from .invariance import run_boundary_inward, run_forward_invariance
from .lipschitz import run_augmented as run_augmented_lipschitz
from .lipschitz import run_physical as run_physical_lipschitz
from .mass_conservation import run as run_mass_conservation
from .nonnegative_rate import run as run_rate_nonnegativity
from .zero_at_depletion import run as run_zero_at_depletion


PROFILES = frozenset(("basic", "physical", "robust", "analysis", "all"))
_BASIC = frozenset(PROFILES)
_PHYSICAL = frozenset(("physical", "robust", "analysis", "all"))
_ROBUST = frozenset(("robust", "all"))
_ANALYSIS = frozenset(("analysis", "all"))
_STAGE_ORDER = {stage: index for index, stage in enumerate(Stage)}


def _placeholder(message: str):
    def run(
        context: AnalysisContext,
        _dependencies: Mapping[str, CheckResult],
    ) -> Finding:
        return Finding(context.case.name, Verdict.UNKNOWN, message)

    return run


CHECK_REGISTRY = (
    CheckSpec(
        "atom_conservation",
        "Atom conservation",
        Stage.CHEMISTRY,
        CheckScope.REACTION,
        (),
        True,
        _BASIC,
        run_atom_conservation,
    ),
    CheckSpec(
        "mass_conservation",
        "Mass conservation",
        Stage.CHEMISTRY,
        CheckScope.REACTION,
        (),
        True,
        _BASIC,
        run_mass_conservation,
    ),
    CheckSpec(
        "physical_rate_definedness",
        "Rate definedness",
        Stage.PHYSICAL,
        CheckScope.REACTION,
        ("atom_conservation", "mass_conservation"),
        True,
        _PHYSICAL,
        run_physical_definedness,
    ),
    CheckSpec(
        "rate_nonnegativity",
        "Rate non-negativity",
        Stage.PHYSICAL,
        CheckScope.REACTION,
        ("physical_rate_definedness",),
        True,
        _PHYSICAL,
        run_rate_nonnegativity,
    ),
    CheckSpec(
        "physical_lipschitz",
        "Lipschitz continuity",
        Stage.PHYSICAL,
        CheckScope.REACTION,
        ("physical_rate_definedness",),
        True,
        _PHYSICAL,
        run_physical_lipschitz,
    ),
    CheckSpec(
        "zero_at_depletion",
        "Zero rate at depletion",
        Stage.PHYSICAL,
        CheckScope.REACTION,
        ("physical_rate_definedness",),
        True,
        _PHYSICAL,
        run_zero_at_depletion,
    ),
    CheckSpec(
        "physical_boundary_inward",
        "Physical boundary inward",
        Stage.PHYSICAL,
        CheckScope.CASE,
        ("rate_nonnegativity", "zero_at_depletion"),
        True,
        _PHYSICAL,
        run_boundary_inward,
    ),
    CheckSpec(
        "forward_invariance",
        "Forward invariance",
        Stage.PHYSICAL,
        CheckScope.CASE,
        ("physical_boundary_inward", "physical_lipschitz"),
        True,
        _PHYSICAL,
        run_forward_invariance,
    ),
    CheckSpec(
        "augmented_rate_definedness",
        "Augmented rate definedness",
        Stage.AUGMENTED,
        CheckScope.REACTION,
        ("atom_conservation", "mass_conservation"),
        True,
        _ROBUST,
        run_augmented_definedness,
    ),
    CheckSpec(
        "augmented_lipschitz",
        "Augmented Lipschitz continuity",
        Stage.AUGMENTED,
        CheckScope.REACTION,
        ("augmented_rate_definedness",),
        True,
        _ROBUST,
        run_augmented_lipschitz,
    ),
    CheckSpec(
        "negative_side_nonrepulsion",
        "Negative-side non-repulsion",
        Stage.AUGMENTED,
        CheckScope.CASE,
        ("augmented_lipschitz",),
        True,
        _ROBUST,
        _placeholder("Negative-side non-repulsion is scheduled for Phase 7."),
    ),
    CheckSpec(
        "conserved_quantities",
        "Conserved quantities",
        Stage.ANALYSIS,
        CheckScope.CASE,
        ("atom_conservation", "mass_conservation"),
        False,
        _ANALYSIS,
        _placeholder("Compact conserved quantities are scheduled for Phase 8."),
    ),
    CheckSpec(
        "structural_faces",
        "Structural faces",
        Stage.ANALYSIS,
        CheckScope.CASE,
        ("atom_conservation", "mass_conservation"),
        False,
        _ANALYSIS,
        _placeholder("Structural-face analysis is scheduled for Phase 8."),
    ),
    CheckSpec(
        "steady_state_equations",
        "Steady-state equations",
        Stage.ANALYSIS,
        CheckScope.CASE,
        ("atom_conservation", "mass_conservation"),
        False,
        _ANALYSIS,
        _placeholder("Steady-state equations are scheduled for Phase 8."),
    ),
)


def validate_registry(
    registry: Iterable[CheckSpec] = CHECK_REGISTRY,
) -> tuple[CheckSpec, ...]:
    """Validate and topologically order a registry, preserving tie order."""

    specs = tuple(registry)
    by_id = {spec.id: spec for spec in specs}
    if len(by_id) != len(specs):
        raise ValueError("Registered check ids must be unique.")
    for spec in specs:
        unknown_profiles = spec.profiles - PROFILES
        if unknown_profiles:
            raise ValueError(
                f"Check '{spec.id}' has unknown profiles: "
                + ", ".join(sorted(unknown_profiles))
                + "."
            )
        if len(spec.requires) != len(set(spec.requires)):
            raise ValueError(f"Check '{spec.id}' repeats a dependency.")
        missing = set(spec.requires) - set(by_id)
        if missing:
            raise ValueError(
                f"Check '{spec.id}' has unknown dependencies: "
                + ", ".join(sorted(missing))
                + "."
            )
        for required in spec.requires:
            if _STAGE_ORDER[by_id[required].stage] > _STAGE_ORDER[spec.stage]:
                raise ValueError(
                    f"Check '{spec.id}' depends on a later stage '{required}'."
                )

    ordered: list[CheckSpec] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(check_id: str) -> None:
        if check_id in visiting:
            raise ValueError(f"Check dependency cycle includes '{check_id}'.")
        if check_id in visited:
            return
        visiting.add(check_id)
        for dependency in by_id[check_id].requires:
            visit(dependency)
        visiting.remove(check_id)
        visited.add(check_id)
        ordered.append(by_id[check_id])

    for spec in sorted(specs, key=lambda item: _STAGE_ORDER[item.stage]):
        visit(spec.id)
    return tuple(ordered)


def plan_checks(
    *,
    profile: str | None = "physical",
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    only: Iterable[str] | None = None,
    registry: Iterable[CheckSpec] = CHECK_REGISTRY,
) -> tuple[CheckSpec, ...]:
    """Expand profiles and dependencies into one deterministic run plan."""

    ordered = validate_registry(registry)
    by_id = {spec.id: spec for spec in ordered}
    included = tuple(include)
    excluded = frozenset(exclude)
    explicit = tuple(only) if only is not None else included
    requested = set(explicit)
    if only is None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown check profile '{profile}'.")
        requested.update(spec.id for spec in ordered if profile in spec.profiles)

    unknown = (requested | set(excluded)) - set(by_id)
    if unknown:
        raise ValueError("Unknown checks: " + ", ".join(sorted(unknown)) + ".")
    overlap = set(explicit) & excluded
    if overlap:
        raise ValueError(
            "Checks cannot be both included and excluded: "
            + ", ".join(sorted(overlap))
            + "."
        )

    def closure(check_ids: Iterable[str]) -> set[str]:
        expanded: set[str] = set()

        def add(check_id: str) -> None:
            if check_id in expanded:
                return
            expanded.add(check_id)
            for dependency in by_id[check_id].requires:
                add(dependency)

        for check_id in check_ids:
            add(check_id)
        return expanded

    explicit_dependencies = closure(explicit)
    conflict = explicit_dependencies & excluded
    if conflict:
        raise ValueError(
            "Explicitly selected checks require excluded checks: "
            + ", ".join(sorted(conflict))
            + "."
        )

    selected = closure(requested) - excluded
    changed = True
    while changed:
        changed = False
        for check_id in tuple(selected):
            if any(required not in selected for required in by_id[check_id].requires):
                selected.remove(check_id)
                changed = True
    return tuple(spec for spec in ordered if spec.id in selected)


validate_registry()


__all__ = ("CHECK_REGISTRY", "PROFILES", "plan_checks", "validate_registry")
