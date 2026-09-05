"""Solver-relevant summaries of the differential profile."""

from collections import Counter

import sympy as sp

from ..proof import SurfaceLocation, profile_differential
from ..results import Evidence, Finding, Verdict


def _metric(value):
    return "unresolved" if value is None else str(sp.N(value, 6))


def _summary(profile):
    physical = profile.physical
    regularity = Counter(rate.regularity.value for rate in physical.rates)
    regularity_text = ", ".join(f"{count} {name}" for name, count in sorted(regularity.items()))
    surfaces = [
        sum(
            surface.location is not SurfaceLocation.EXCLUDED
            for rate in domain.rates
            for surface in rate.surfaces
        )
        for domain in (physical, profile.augmented)
    ]
    return (
        f"Rank {profile.stoichiometric_rank}; {profile.source_equivalent_fluxes} source-equivalent fluxes. "
        f"Physical regularity: {regularity_text}. "
        f"Scaled Jacobian bound: {_metric(physical.source_jacobian.infinity_norm_upper)}; "
        f"active-mode bound: {_metric(physical.reduced_jacobian.spectral_radius_upper)}; "
        f"Jacobian variation: {_metric(physical.jacobian_variation_upper)}. "
        f"Active surfaces: {surfaces[0]} physical, {surfaces[1]} augmented."
    )


def run(context, _dependencies):
    symbols = context.case.symbols
    profile = profile_differential(
        analyzer=context.expression_analyzer,
        reactions=context.case.reactions,
        stoichiometry=context.stoichiometry,
        concentration_symbols=tuple(symbols.concentrations.values()),
        operating_symbols=(symbols.temperature, symbols.pressure),
        physical_domain=context.physical_domain,
        augmented_domain=context.augmented_domain,
    )
    complete = all(
        domain.interaction.complete
        and domain.source_jacobian.complete
        and all(
            item.absolute_upper is not None for rate in domain.rates for item in rate.derivatives
        )
        for domain in (profile.physical, profile.augmented)
    )
    return Finding(
        context.case.name,
        Verdict.PASS if complete else Verdict.UNKNOWN,
        _summary(profile),
        Evidence("differential_solver_profile", vars(profile)),
    )
