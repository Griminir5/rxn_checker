"""Adapter for the nonblocking differential solver profile."""

from collections import Counter
from collections.abc import Mapping

import sympy as sp

from ..context import AnalysisContext
from ..proof import (
    DifferentialSolverProfile,
    DomainDifferentialProfile,
    MatrixEnvelope,
    RateDifferentialProfile,
    SurfaceLocation,
    profile_differential,
)
from ..results import CheckResult, Evidence, Finding, Verdict


def _approximate(value: sp.Expr | None) -> str | None:
    if value is None:
        return None
    try:
        return str(sp.N(value, 6))
    except (TypeError, ValueError):
        return None


def _bound(
    value: sp.Expr | None,
    *,
    complete: bool,
    exact: bool = False,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "value": value,
        "approximate_value": _approximate(value),
        "exact_enclosure": exact,
        "complete": complete and value is not None,
        "reason": reason,
    }


def _entry_data(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        **entry,
        "absolute_bound": _bound(
            entry.get("absolute_upper"),
            complete=bool(entry.get("complete")),
            exact=bool(entry.get("exact_enclosure")),
            reason=entry.get("reason"),
        ),
    }


def _matrix_data(matrix: MatrixEnvelope) -> dict[str, object]:
    return {
        "shape": matrix.shape,
        "structural_nonzeros": matrix.structural_nonzeros,
        "density": matrix.density,
        "rows": matrix.row_labels,
        "columns": matrix.column_labels,
        "entries": tuple(_entry_data(item) for item in matrix.entries),
        "infinity_norm_upper": _bound(
            matrix.infinity_norm_upper,
            complete=matrix.complete,
            reason=matrix.reason,
        ),
        "logarithmic_norm_upper": _bound(
            matrix.logarithmic_norm_upper,
            complete=matrix.complete,
            reason=matrix.reason,
        ),
        "active_mode_magnitude_upper": _bound(
            matrix.spectral_radius_upper,
            complete=matrix.complete,
            reason=matrix.reason,
        ),
        "complete": matrix.complete,
        "reason": matrix.reason,
        "metadata": matrix.metadata or {},
    }


def _rate_data(rate: RateDifferentialProfile) -> dict[str, object]:
    inverse = (
        1 / rate.self_feedback_absolute_upper
        if rate.self_feedback_absolute_upper not in (None, 0)
        else None
    )
    return {
        "regularity": rate.regularity,
        "reduced_expression": rate.reduced_expression,
        "branch_reductions": rate.branch_reductions,
        "surfaces": tuple(
            {
                "kind": item.kind,
                "expression": item.expression,
                "source": item.source,
                "location": item.location,
                "reason": item.reason,
            }
            for item in rate.surfaces
        ),
        "derivatives": tuple(
            {
                "variable": item.variable,
                "derivative": item.derivative,
                "lower": item.lower,
                "upper": item.upper,
                "absolute_bound": _bound(
                    item.absolute_upper,
                    complete=item.absolute_upper is not None,
                    exact=item.exact_enclosure,
                    reason=item.reason,
                ),
                "signed": item.signed,
            }
            for item in rate.derivatives
        ),
        "self_feedback": {
            "expression": rate.self_feedback_expression,
            "lower": rate.self_feedback_lower,
            "upper": rate.self_feedback_upper,
            "kind": rate.self_feedback_kind,
            "absolute_bound": _bound(
                rate.self_feedback_absolute_upper,
                complete=rate.self_feedback_absolute_upper is not None,
            ),
            "inverse_self_feedback_magnitude": _bound(
                inverse,
                complete=inverse is not None,
                reason="Lower bound on a reaction-direction linear timescale; not an IDAS step size.",
            ),
        },
        "source_jacobian_contribution": _bound(
            rate.source_jacobian_contribution,
            complete=rate.source_jacobian_contribution is not None,
        ),
        "curvature": {
            "status": rate.curvature_status,
            "source_jacobian_variation_contribution": _bound(
                rate.curvature_contribution,
                complete=rate.curvature_contribution is not None,
            ),
            "hessian_entries": rate.hessian,
        },
    }


def _domain_data(profile: DomainDifferentialProfile) -> dict[str, object]:
    regularity = Counter(rate.regularity.value for rate in profile.rates)
    active_surfaces = tuple(
        item
        for rate in profile.rates
        for item in rate.surfaces
        if item.location is not SurfaceLocation.EXCLUDED
    )
    reductions = sum(len(rate.branch_reductions) for rate in profile.rates)
    fast = sorted(
        profile.rates,
        key=lambda rate: (-_number(rate.source_jacobian_contribution), rate.rate_id),
    )
    variation = sorted(
        profile.rates,
        key=lambda rate: (-_number(rate.curvature_contribution), rate.rate_id),
    )
    return {
        "regularity_summary": dict(regularity),
        "branch_reduction_count": reductions,
        "active_surface_count": len(active_surfaces),
        "active_surfaces_by_location": dict(
            Counter(item.location.value for item in active_surfaces)
        ),
        "concentration_scales": dict(profile.concentration_scales),
        "rates": {rate.rate_id: _rate_data(rate) for rate in profile.rates},
        "largest_fast_mode_contributors": tuple(
            {
                "rate_id": rate.rate_id,
                "bound": _bound(
                    rate.source_jacobian_contribution,
                    complete=rate.source_jacobian_contribution is not None,
                ),
            }
            for rate in fast[:5]
        ),
        "largest_jacobian_variation_contributors": tuple(
            {
                "rate_id": rate.rate_id,
                "bound": _bound(
                    rate.curvature_contribution,
                    complete=rate.curvature_contribution is not None,
                ),
            }
            for rate in variation[:5]
        ),
        "interaction_matrix": _matrix_data(profile.interaction),
        "source_jacobian": _matrix_data(profile.source_jacobian),
        "reduced_jacobian": _matrix_data(profile.reduced_jacobian),
        "operating_coupling": _matrix_data(profile.operating_coupling),
        "ida_alpha_dominance_threshold": _bound(
            profile.ida_alpha_dominance_threshold,
            complete=profile.source_jacobian.complete,
            reason=(
                "Above this value, the isolated domain-scaled kinetic Schur complement "
                "is certified strictly row diagonally dominant; this is not a step size."
            ),
        ),
        "scaled_source_jacobian_variation_bound": _bound(
            profile.jacobian_variation_upper,
            complete=profile.jacobian_variation_upper is not None,
            reason=(
                "Incomplete Hessian bounds prevent a global value."
                if profile.jacobian_variation_upper is None
                else None
            ),
        ),
        "hessian_truncated": profile.hessian_truncated,
    }


def _evidence(profile: DifferentialSolverProfile) -> dict[str, object]:
    return {
        "target": "kinetic concentration-space subsystem",
        "solver_context": {
            "form": "F(t, y, y_dot) = 0",
            "kinetic_schur_complement": "alpha*I - S*dr_dc",
            "profile_label": "kinetic concentration-space differential profile",
            "full_daetools_jacobian": False,
        },
        "network": {
            "declared_reactions": profile.declared_reactions,
            "source_equivalent_fluxes": profile.source_equivalent_fluxes,
            "stoichiometric_rank": profile.stoichiometric_rank,
            "stoichiometric_basis": profile.stoichiometric_basis,
        },
        "physical": _domain_data(profile.physical),
        "augmented": _domain_data(profile.augmented),
    }


def _number(value: sp.Expr | None) -> float:
    try:
        return float(sp.N(value, 8)) if value is not None else float("-inf")
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


def _metric(value: sp.Expr | None) -> str:
    approximate = _approximate(value)
    return approximate if approximate is not None else "unresolved"


def _summary(profile: DifferentialSolverProfile) -> str:
    physical = profile.physical
    augmented = profile.augmented
    regularity = Counter(rate.regularity.value for rate in physical.rates)
    regularity_text = ", ".join(
        f"{count} {name}" for name, count in sorted(regularity.items())
    )
    feedback = sorted(
        physical.rates,
        key=lambda rate: (-_number(rate.self_feedback_absolute_upper), rate.rate_id),
    )[:3]
    contributors = sorted(
        physical.rates,
        key=lambda rate: (-_number(rate.source_jacobian_contribution), rate.rate_id),
    )[:3]
    feedback_text = ", ".join(
        f"{rate.rate_id} ({rate.self_feedback_kind.value}, "
        f"{_metric(rate.self_feedback_absolute_upper)})"
        for rate in feedback
    )
    contributor_text = ", ".join(
        f"{rate.rate_id} ({_metric(rate.source_jacobian_contribution)})"
        for rate in contributors
    )
    physical_surfaces = sum(
        item.location is not SurfaceLocation.EXCLUDED
        for rate in physical.rates
        for item in rate.surfaces
    )
    augmented_surfaces = sum(
        item.location is not SurfaceLocation.EXCLUDED
        for rate in augmented.rates
        for item in rate.surfaces
    )
    incomplete = []
    for name, domain in (("physical", physical), ("augmented", augmented)):
        if not (domain.interaction.complete and domain.source_jacobian.complete):
            incomplete.append(f"{name} first-order matrices")
        if domain.hessian_truncated or domain.jacobian_variation_upper is None:
            incomplete.append(f"{name} curvature")
    return (
        f"Kinetic source rank {profile.stoichiometric_rank}; "
        f"{profile.source_equivalent_fluxes} source-equivalent fluxes. Physical "
        f"regularity: {regularity_text}; {physical_surfaces} active surfaces. "
        f"Domain-scaled source-Jacobian bound "
        f"{_metric(physical.source_jacobian.infinity_norm_upper)}; active-mode "
        f"magnitude upper bound {_metric(physical.reduced_jacobian.spectral_radius_upper)}; "
        f"IDA alpha diagonal-dominance threshold "
        f"{_metric(physical.ida_alpha_dominance_threshold)}; scaled Jacobian-variation "
        f"bound {_metric(physical.jacobian_variation_upper)}. Strongest self-feedback: "
        f"{feedback_text}. Largest fast-mode contributors: {contributor_text}. "
        f"Augmented domain has {augmented_surfaces - physical_surfaces:+d} active "
        f"surfaces relative to physical. Incomplete sections: "
        f"{', '.join(incomplete) if incomplete else 'none'}."
    )


def _complete(profile: DifferentialSolverProfile) -> bool:
    return all(
        domain.interaction.complete
        and domain.source_jacobian.complete
        and all(
            derivative.absolute_upper is not None
            for rate in domain.rates
            for derivative in rate.derivatives
        )
        for domain in (profile.physical, profile.augmented)
    )


def run(
    context: AnalysisContext,
    _dependencies: Mapping[str, CheckResult],
) -> Finding:
    profile = profile_differential(
        analyzer=context.expression_analyzer,
        reactions=context.case.reactions,
        stoichiometry=context.stoichiometry,
        concentration_symbols=tuple(context.case.symbols.concentrations.values()),
        operating_symbols=(context.case.symbols.temperature, context.case.symbols.pressure),
        physical_domain=context.physical_domain,
        augmented_domain=context.augmented_domain,
    )
    return Finding(
        context.case.name,
        Verdict.PASS if _complete(profile) else Verdict.UNKNOWN,
        _summary(profile),
        Evidence("differential_solver_profile", _evidence(profile)),
    )


__all__ = ("run",)
