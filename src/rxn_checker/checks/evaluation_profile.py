"""Report evaluation work and common-subexpression reuse."""

from ..proof import profile_evaluation
from ..results import Evidence, Finding, Verdict


def run(context, _dependencies):
    symbols = context.case.symbols
    profile = profile_evaluation(
        context.case.reactions,
        context.stoichiometry,
        symbols.concentration_symbols,
        symbols.parameter_symbols,
    )
    summaries = []
    unsupported = set()
    for label, view in (
        ("declared rates", profile.declared),
        ("source-equivalent fluxes", profile.source_equivalent),
    ):
        summaries.append(
            f"{len(view.outputs)} {label}: {view.raw.total_operations} operations/cell "
            f"({view.raw.transcendental_operations} transcendental, {view.raw.switch_operations} switch); "
            f"{view.cse.total_operations} after CSE, {view.cse.temporary_count} temporaries."
        )
        unsupported.update(
            expression
            for stats in view.outputs.values()
            for expression in stats.unsupported_subexpressions
        )
    expensive = sorted(
        profile.source_equivalent.outputs.items(),
        key=lambda item: (-item[1].total_operations, item[0]),
    )[:3]
    names = ", ".join(f"{name} ({stats.total_operations})" for name, stats in expensive)
    summaries.append(f"Most expensive: {names}. Unsupported operations: {len(unsupported)}.")
    return Finding(
        context.case.name,
        Verdict.UNKNOWN if unsupported else Verdict.PASS,
        " ".join(summaries),
        Evidence("evaluation_profile", vars(profile)),
    )
