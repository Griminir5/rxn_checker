"""Adapter for the nonblocking DAETools-oriented evaluation profile."""

from collections import Counter, defaultdict
from collections.abc import Mapping

import sympy as sp

from ..context import AnalysisContext
from ..proof import (
    CSEStats,
    EvaluationProfile,
    ExpressionStats,
    OPERATION_ORDER,
    profile_evaluation,
)
from ..results import CheckResult, Evidence, Finding, Verdict


def _operation_data(stats: ExpressionStats | CSEStats) -> dict[str, object]:
    return {
        "operations": dict(stats.operations),
        "total_operations": stats.total_operations,
        "transcendental_operations": stats.transcendental_operations,
        "switch_operations": stats.switch_operations,
    }


def _raw_data(
    outputs: tuple[tuple[str, ExpressionStats], ...],
) -> dict[str, object]:
    operations: Counter[str] = Counter()
    for _output_id, stats in outputs:
        operations.update(dict(stats.operations))
    histogram = {name: operations[name] for name in OPERATION_ORDER}
    total = sum(histogram.values())
    return {
        "operations": histogram,
        "total_operations": total,
        "transcendental_operations": sum(
            stats.transcendental_operations for _, stats in outputs
        ),
        "switch_operations": sum(stats.switch_operations for _, stats in outputs),
        "tree_nodes": sum(stats.tree_nodes for _, stats in outputs),
        "unique_nodes": sum(stats.unique_nodes for _, stats in outputs),
        "maximum_depth": max((stats.depth for _, stats in outputs), default=0),
        "dependency_weighted_ad_work": sum(
            stats.ad_work for _, stats in outputs
        ),
        "piecewise_branches": sum(
            stats.piecewise_branches for _, stats in outputs
        ),
    }


def _cse_data(stats: CSEStats, raw_operations: int) -> dict[str, object]:
    data = _operation_data(stats)
    saving = raw_operations - stats.total_operations
    data.update(
        {
            "operation_reduction": saving,
            "shareable_fraction": saving / raw_operations if raw_operations else 0.0,
            "temporary_count": stats.temporary_count,
            "peak_live_temporaries": stats.peak_live_temporaries,
        }
    )
    return data


def _residual_data(
    outputs: tuple[tuple[str, ExpressionStats], ...], source_nnz: int
) -> dict[str, int]:
    input_entries = sum(
        stats.structural_jacobian_entries for _, stats in outputs
    )
    rate_equations = len(outputs)
    return {
        "rate_equations": rate_equations,
        "rate_input_structural_entries": input_entries,
        "stoichiometric_source_links": source_nnz,
        "total_kinetics_block_structural_entries": (
            rate_equations + input_entries + source_nnz
        ),
        "maximum_rate_dependency_width": max(
            (stats.structural_jacobian_entries for _, stats in outputs),
            default=0,
        ),
    }


def _output_data(stats: ExpressionStats, local: CSEStats) -> dict[str, object]:
    raw = _operation_data(stats)
    raw.update(
        {
            "tree_nodes": stats.tree_nodes,
            "unique_nodes": stats.unique_nodes,
            "maximum_depth": stats.depth,
            "piecewise_branches": stats.piecewise_branches,
        }
    )
    return {
        "raw": raw,
        "local_cse": _cse_data(local, stats.total_operations),
        "dependencies": {
            "concentrations": stats.concentration_dependencies,
            "temperature_pressure": stats.operating_dependencies,
            "all_dae_inputs": stats.dae_dependencies,
            "structural_jacobian_entries": stats.structural_jacobian_entries,
        },
        "dependency_weighted_ad_work": stats.ad_work,
        "unsupported_functions": stats.unsupported_functions,
    }


def _groups(profile: EvaluationProfile) -> tuple[dict[str, object], ...]:
    rendered = []
    for group in profile.flux_groups:
        rendered.append(
            {
                "id": group["id"],
                "stoichiometry": group["stoichiometry"],
                "members": tuple(
                    {"reaction_id": reaction_id, "coefficient": coefficient}
                    for reaction_id, coefficient in group["members"]
                ),
                "expression": group["expression"],
            }
        )
    return tuple(rendered)


def _unsupported(profile: EvaluationProfile) -> tuple[dict[str, object], ...]:
    affected: defaultdict[tuple[str, sp.Expr], list[str]] = defaultdict(list)
    for view, outputs in (
        ("declared", profile.declared_outputs),
        ("source_equivalent", profile.flux_outputs),
    ):
        for output_id, stats in outputs:
            for expression in stats.unsupported_subexpressions:
                name = (
                    "Pow"
                    if isinstance(expression, sp.Pow)
                    else expression.func.__name__
                )
                affected[(name, expression)].append(f"{view}:{output_id}")
    return tuple(
        {
            "function_name": name,
            "decisive_subexpression": expression,
            "affected_outputs": tuple(outputs),
        }
        for (name, expression), outputs in affected.items()
    )


def _evidence(profile: EvaluationProfile) -> dict[str, object]:
    declared_raw = _raw_data(profile.declared_outputs)
    flux_raw = _raw_data(profile.flux_outputs)
    declared_residual = _residual_data(
        profile.declared_outputs, profile.declared_source_nnz
    )
    flux_residual = _residual_data(profile.flux_outputs, profile.flux_source_nnz)
    declared_local = dict(profile.declared_local_cse)
    flux_local = dict(profile.flux_local_cse)
    return {
        "target": "DAETools mathematical expression tree",
        "units": "operations per cell per residual evaluation",
        "declared": {
            "output_count": len(profile.declared_outputs),
            "raw": declared_raw,
            "global_cse": _cse_data(
                profile.declared_cse, int(declared_raw["total_operations"])
            ),
            "source_nnz": profile.declared_source_nnz,
            "residual_jacobian_nnz": declared_residual[
                "total_kinetics_block_structural_entries"
            ],
            "residual_structure": declared_residual,
        },
        "source_equivalent": {
            "output_count": len(profile.flux_outputs),
            "groups": _groups(profile),
            "raw": flux_raw,
            "global_cse": _cse_data(
                profile.flux_cse, int(flux_raw["total_operations"])
            ),
            "source_nnz": profile.flux_source_nnz,
            "residual_jacobian_nnz": flux_residual[
                "total_kinetics_block_structural_entries"
            ],
            "residual_structure": flux_residual,
        },
        "outputs": {
            "declared": {
                output_id: _output_data(stats, declared_local[output_id])
                for output_id, stats in profile.declared_outputs
            },
            "source_equivalent": {
                output_id: _output_data(stats, flux_local[output_id])
                for output_id, stats in profile.flux_outputs
            },
        },
        "shared_terms": tuple(
            {
                "expression": term.expression,
                "operation_count": term.operations,
                "occurrences": term.occurrences,
                "outputs": term.outputs,
                "reuse": "cross-rate" if len(term.outputs) > 1 else "within-rate",
                "individual_potential_saving": term.estimated_saved_operations,
            }
            for term in profile.shared_terms
        ),
        "unsupported_operations": _unsupported(profile),
    }


def _summary(profile: EvaluationProfile) -> str:
    declared_raw = sum(
        stats.total_operations for _, stats in profile.declared_outputs
    )
    flux_raw = sum(stats.total_operations for _, stats in profile.flux_outputs)
    declared_trans = sum(
        stats.transcendental_operations for _, stats in profile.declared_outputs
    )
    declared_switch = sum(
        stats.switch_operations for _, stats in profile.declared_outputs
    )
    flux_trans = sum(
        stats.transcendental_operations for _, stats in profile.flux_outputs
    )
    flux_switch = sum(stats.switch_operations for _, stats in profile.flux_outputs)
    residual = _residual_data(profile.flux_outputs, profile.flux_source_nnz)
    expensive = sorted(
        profile.flux_outputs,
        key=lambda item: (-item[1].total_operations, item[0]),
    )[:3]
    expensive_text = ", ".join(
        f"{output_id} ({stats.total_operations})" for output_id, stats in expensive
    )
    unsupported_count = len(_unsupported(profile))
    shared_text = ", ".join(
        f"{_short_expression(term.expression)} ({term.estimated_saved_operations})"
        for term in profile.shared_terms[:5]
    ) or "none"
    return (
        f"{len(profile.declared_outputs)} declared rates: {declared_raw} operations/cell "
        f"({declared_trans} transcendental, {declared_switch} switch); global CSE "
        f"leaves {profile.declared_cse.total_operations} with "
        f"{profile.declared_cse.temporary_count} temporaries, at most "
        f"{profile.declared_cse.peak_live_temporaries} live. "
        f"{len(profile.flux_outputs)} source-equivalent fluxes: {flux_raw} raw "
        f"operations ({flux_trans} transcendental, {flux_switch} switch); global "
        f"CSE leaves {profile.flux_cse.total_operations} with "
        f"{profile.flux_cse.temporary_count} temporaries, at most "
        f"{profile.flux_cse.peak_live_temporaries} live. Kinetics residual "
        f"structure: {residual['rate_equations']} rate equations, "
        f"{residual['rate_input_structural_entries']} rate-input links, "
        f"{residual['stoichiometric_source_links']} stoichiometric links, "
        f"{residual['total_kinetics_block_structural_entries']} total. Most "
        f"expensive source-equivalent outputs: {expensive_text}. "
        f"Largest reuse opportunities (individual, non-additive savings): "
        f"{shared_text}. Unsupported DAETools operations: {unsupported_count}."
    )


def _short_expression(expression: sp.Expr, limit: int = 64) -> str:
    text = sp.sstr(expression)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run(
    context: AnalysisContext,
    _dependencies: Mapping[str, CheckResult],
) -> Finding:
    """Return a comparative, nonblocking evaluation profile for one case."""

    profile = profile_evaluation(
        context.case.reactions,
        context.stoichiometry,
        context.case.symbols.concentration_symbols,
        context.case.symbols.parameter_symbols,
    )
    unsupported = any(
        stats.unsupported_functions
        for outputs in (profile.declared_outputs, profile.flux_outputs)
        for _, stats in outputs
    )
    return Finding(
        context.case.name,
        Verdict.UNKNOWN if unsupported else Verdict.PASS,
        _summary(profile),
        Evidence("evaluation_profile", _evidence(profile)),
    )


__all__ = ("run",)
