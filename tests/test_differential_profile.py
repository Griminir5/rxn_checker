"""Differential solver profile contracts."""

import sympy as sp

from rxn_checker import (
    Case,
    CaseSymbols,
    ConcentrationModel,
    DomainSpec,
    Interval,
    Reaction,
    Verdict,
)
from rxn_checker.checks import plan_checks, run_checks
from rxn_checker.domain import ConcentrationDomain, DomainKind
from rxn_checker.proof import (
    ExpressionAnalyzer,
    FeedbackKind,
    Regularity,
    SurfaceLocation,
    profile_differential,
)
from rxn_checker.species import PROPERTY_REGISTRY


A, B, C, T, P = sp.symbols("A B C T P", real=True)


def _domain(kind, limits, parameters=None):
    return ConcentrationDomain(
        kind,
        {symbol: Interval(*bounds) for symbol, bounds in limits.items()},
        {
            symbol: Interval(*bounds)
            for symbol, bounds in (parameters or {}).items()
        },
    )


def _profile(reactions, stoichiometry, physical, augmented=None, operating=()):
    symbols = tuple(physical.intervals)
    return profile_differential(
        analyzer=ExpressionAnalyzer(),
        reactions=reactions,
        stoichiometry=sp.Matrix(stoichiometry),
        concentration_symbols=symbols,
        operating_symbols=operating,
        physical_domain=physical,
        augmented_domain=augmented or ConcentrationDomain(
            DomainKind.AUGMENTED,
            physical.intervals,
            physical.parameter_intervals,
        ),
    )


def test_linear_decay_feedback_jacobian_and_active_mode_are_exact() -> None:
    reaction = Reaction("linear", {"A": 1}, {"B": 1}, (), 2 * A)
    domain = _domain(DomainKind.PHYSICAL, {A: (0, 3), B: (0, 3)})
    profile = _profile((reaction,), ((-1,), (1,)), domain)
    rate = profile.physical.rates[0]

    assert rate.regularity is Regularity.C11
    assert rate.self_feedback_lower == -2
    assert rate.self_feedback_upper == -2
    assert rate.self_feedback_kind is FeedbackKind.DAMPING
    source = {(item["row"], item["column"]): item for item in profile.physical.source_jacobian.entries}
    assert source[('A', 'A')]["lower"] == -2
    assert source[('B', 'A')]["upper"] == 2
    assert profile.physical.reduced_jacobian.shape == (1, 1)
    assert profile.physical.reduced_jacobian.entries[0]["lower"] == -2
    assert profile.physical.ida_alpha_dominance_threshold == 2


def test_autocatalytic_feedback_uses_the_reaction_direction() -> None:
    reaction = Reaction("auto", {"A": 1, "B": 1}, {"B": 2}, (), A * B)

    def feedback(a_bounds, b_bounds):
        domain = _domain(DomainKind.PHYSICAL, {A: a_bounds, B: b_bounds})
        return _profile((reaction,), ((-1,), (1,)), domain).physical.rates[0]

    amplifying = feedback((3, 4), (1, 2))
    damping = feedback((1, 2), (3, 4))
    mixed = feedback((1, 4), (1, 4))

    assert sp.expand(amplifying.self_feedback_expression) == A - B
    assert amplifying.self_feedback_kind is FeedbackKind.AMPLIFYING
    assert damping.self_feedback_kind is FeedbackKind.DAMPING
    assert mixed.self_feedback_kind is FeedbackKind.MIXED


def test_reversible_directions_are_grouped_without_changing_the_source() -> None:
    reactions = (
        Reaction("pair.r_fw", {"A": 1}, {"B": 1}, (), 3 * A),
        Reaction("pair.r_bw", {"B": 1}, {"A": 1}, (), 2 * B),
    )
    domain = _domain(DomainKind.PHYSICAL, {A: (0, 3), B: (0, 3)})
    profile = _profile(reactions, ((-1, 1), (1, -1)), domain)
    rate = profile.physical.rates[0]

    assert profile.source_equivalent_fluxes == 1
    assert rate.rate_id == "pair.r_net"
    assert sp.expand(rate.reduced_expression) == 3 * A - 2 * B
    assert profile.stoichiometric_rank == 1


def test_interaction_graph_retains_one_way_coupling() -> None:
    reactions = (
        Reaction("first", {"A": 1}, {"B": 1}, (), A),
        Reaction("second", {"B": 1}, {"C": 1}, (), B),
    )
    domain = _domain(
        DomainKind.PHYSICAL, {A: (0, 3), B: (0, 3), C: (0, 3)}
    )
    profile = _profile(reactions, ((-1, 0), (1, -1), (0, 1)), domain)
    entries = {
        (item["row"], item["column"])
        for item in profile.physical.interaction.entries
    }

    assert ("second", "first") in entries
    assert ("first", "second") not in entries
    assert profile.physical.interaction.metadata["one_way_component_edges"]


def test_branch_reduction_separates_physical_and_augmented_domains() -> None:
    reaction = Reaction("clamp", {"A": 1}, {"B": 1}, (), sp.Max(A, 0) * B)
    physical = _domain(DomainKind.PHYSICAL, {A: (0, 3), B: (0, 3)})
    augmented = _domain(DomainKind.AUGMENTED, {A: (-1, 3), B: (-1, 3)})
    profile = _profile((reaction,), ((-1,), (1,)), physical, augmented)
    smooth, switched = profile.physical.rates[0], profile.augmented.rates[0]

    assert smooth.reduced_expression == A * B
    assert smooth.branch_reductions
    assert smooth.regularity is Regularity.C11
    assert switched.regularity is Regularity.PIECEWISE_C1
    assert switched.surfaces[0].location is SurfaceLocation.INTERIOR
    assert not any(item.signed for item in switched.derivatives)


def test_boundary_derivative_singularity_is_reported_without_raising() -> None:
    reaction = Reaction("root", {"A": 1}, {"B": 1}, (), sp.sqrt(A))
    physical = _domain(DomainKind.PHYSICAL, {A: (0, 1), B: (0, 1)})
    positive = _domain(DomainKind.AUGMENTED, {A: (1, 2), B: (0, 1)})
    profile = _profile((reaction,), ((-1,), (1,)), physical, positive)

    boundary = profile.physical.rates[0]
    assert boundary.regularity is Regularity.CONTINUOUS
    assert boundary.derivatives[0].absolute_upper is None
    assert boundary.surfaces[0].location is SurfaceLocation.BOUNDARY
    assert profile.augmented.rates[0].derivatives[0].absolute_upper == sp.Rational(1, 2)


def test_operating_coupling_and_scaled_curvature_are_separate() -> None:
    reaction = Reaction(
        "operating", {"A": 1}, {"B": 1}, (), A**2 * sp.exp(-1 / T) * P
    )
    parameters = {T: (2, 4), P: (1, 2)}
    physical = _domain(
        DomainKind.PHYSICAL, {A: (0, 3), B: (0, 3)}, parameters
    )
    profile = _profile(
        (reaction,), ((-1,), (1,)), physical, operating=(T, P)
    )
    rate = profile.physical.rates[0]

    assert {item.variable for item in rate.derivatives} == {A, T, P}
    assert profile.physical.operating_coupling.structural_nonzeros == 4
    assert profile.physical.operating_coupling.metadata["column_bounds"]["T"] > 0
    assert any(item["left"] == A and item["right"] == A for item in rate.hessian)


def test_unsupported_symbolic_derivative_keeps_partial_results() -> None:
    unknown = sp.Function("unknown")(A)
    reaction = Reaction("unsupported", {"A": 1}, {"B": 1}, (), unknown)
    domain = _domain(DomainKind.PHYSICAL, {A: (0, 1), B: (0, 1)})
    profile = _profile((reaction,), ((-1,), (1,)), domain)

    assert profile.physical.rates[0].regularity is Regularity.UNKNOWN
    assert not profile.physical.source_jacobian.complete
    assert profile.physical.source_jacobian.structural_nonzeros == 2


def test_check_is_nonblocking_and_selected_only_by_analysis_profiles() -> None:
    symbols = CaseSymbols.for_species(("Aye", "Bee"))
    aye = symbols.concentration("Aye")
    reaction = Reaction("linear", {"Aye": 1}, {"Bee": 1}, (), aye)
    domain = DomainSpec(
        symbols,
        ConcentrationModel.INDEPENDENT,
        {symbol: 3 for symbol in symbols.concentration_symbols},
        {symbol: -1 for symbol in symbols.concentration_symbols},
        {
            symbols.temperature: Interval(300, 400),
            symbols.pressure: Interval(1, 2),
        },
    )
    case = Case(
        "linear",
        tuple(PROPERTY_REGISTRY.get_record(item) for item in symbols.species_ids),
        symbols,
        (reaction,),
        domain,
    )
    result = run_checks(case, only=("differential_solver_profile",), debug=True)

    assert result.results["differential_solver_profile"].findings[0].verdict is Verdict.PASS
    assert result.overall is Verdict.PASS
    assert "differential_solver_profile" not in {
        spec.id for spec in plan_checks(profile="physical")
    }
    for name in ("analysis", "all"):
        assert "differential_solver_profile" in {
            spec.id for spec in plan_checks(profile=name)
        }
