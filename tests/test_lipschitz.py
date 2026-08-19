"""Phase 5 scalar and network Lipschitz contracts."""

import sympy as sp

from rxn_checker.domain import (
    ConcentrationDomain,
    DomainKind,
    Interval,
    TotalConstraint,
)
from rxn_checker.proof import (
    ExpressionAnalyzer,
    ProofVerdict,
    derive_network_lipschitz,
)


c, d, temperature = sp.symbols("c d temperature", real=True)


def _domain(
    lower=0,
    upper=3,
    *,
    kind=DomainKind.PHYSICAL,
    parameters=None,
) -> ConcentrationDomain:
    return ConcentrationDomain(
        kind,
        {c: Interval(lower, upper)},
        parameters or {},
    )


def _constant(expression, domain=None):
    result = ExpressionAnalyzer().lipschitz(expression, domain or _domain())
    assert result.verdict is ProofVerdict.PASS
    return result.certificate.constant_bound


def test_elementary_compositional_constants_are_checkable() -> None:
    domain = _domain(0, 3)

    assert _constant(c, domain) == 1
    assert _constant(2 * c, domain) == 2
    assert _constant(c**2, domain) == 6
    assert _constant(sp.Abs(c), domain) == 1
    assert _constant(sp.Max(c, 0), domain) == 1


def test_reciprocal_requires_a_strict_nonzero_margin() -> None:
    failure = ExpressionAnalyzer().lipschitz(1 / c, _domain(0, 2))
    success = ExpressionAnalyzer().lipschitz(1 / c, _domain(1, 2))

    assert failure.verdict is ProofVerdict.FAIL
    assert failure.decisive_subexpression == c
    assert failure.witness[c] == 0
    assert success.verdict is ProofVerdict.PASS
    assert success.certificate.constant_bound == 1
    assert success.certificate.guard_margins[0].margin == 1


def test_total_constraint_can_supply_the_reciprocal_margin() -> None:
    independent = ConcentrationDomain(
        DomainKind.PHYSICAL,
        {c: Interval(0, 3), d: Interval(0, 2)},
        {},
    )
    chamfered = ConcentrationDomain(
        DomainKind.PHYSICAL,
        independent.intervals,
        {},
        (TotalConstraint("gas", (c, d), 1),),
    )
    analyzer = ExpressionAnalyzer()

    assert analyzer.lipschitz(1 / (c + d), independent).verdict is ProofVerdict.FAIL
    result = analyzer.lipschitz(1 / (c + d), chamfered)
    assert result.verdict is ProofVerdict.PASS
    assert result.certificate.constant_bound == 2
    assert result.certificate.guard_margins[0].margin == 1


def test_noninteger_power_and_log_need_an_open_neighbourhood() -> None:
    analyzer = ExpressionAnalyzer()
    domain = _domain(0, 3)

    assert analyzer.lipschitz(sp.sqrt(c), domain).verdict is ProofVerdict.FAIL
    assert analyzer.lipschitz(sp.log(c), domain).verdict is ProofVerdict.FAIL
    regularized = analyzer.lipschitz(sp.sqrt(c**2 + 1), domain)
    assert regularized.verdict is ProofVerdict.PASS
    assert regularized.certificate.guard_margins[0].margin == 1


def test_parameter_bounds_are_uniform_but_not_state_coordinates() -> None:
    domain = _domain(parameters={temperature: Interval(1, 4)})
    result = ExpressionAnalyzer().lipschitz(temperature * c, domain)

    assert result.verdict is ProofVerdict.PASS
    assert result.certificate.constant_bound == 4
    assert result.certificate.active_variables == (c,)
    assert result.certificate.uniform_parameters == (temperature,)


def test_bounded_derivative_fallback_and_discontinuous_unknown() -> None:
    analyzer = ExpressionAnalyzer()
    domain = _domain()

    smooth = analyzer.lipschitz(2 * sp.erf(c), domain)
    discontinuous = analyzer.lipschitz(sp.floor(c), domain)
    assert smooth.verdict is ProofVerdict.PASS
    assert smooth.certificate.constant_bound == 4 / sp.sqrt(sp.pi)
    assert discontinuous.verdict is ProofVerdict.UNKNOWN


def test_analysis_is_cached_by_expression_domain_and_active_variables() -> None:
    analyzer = ExpressionAnalyzer()
    domain = _domain()

    first = analyzer.lipschitz(c**2, domain)
    assert analyzer.lipschitz(c**2, domain) is first


def test_network_constant_is_derived_from_stoichiometry() -> None:
    result = ExpressionAnalyzer().lipschitz(2 * c, _domain())
    certificate = derive_network_lipschitz(
        _domain(),
        ("reactant", "product"),
        sp.ImmutableMatrix([[-1], [1]]),
        (result.certificate,),
    )

    assert certificate.component_bounds == (("reactant", 2), ("product", 2))
    assert certificate.constant_bound == 2


def test_certificate_records_the_selected_domain() -> None:
    physical = ExpressionAnalyzer().lipschitz(c, _domain(kind=DomainKind.PHYSICAL))
    augmented = ExpressionAnalyzer().lipschitz(
        c, _domain(-1, 3, kind=DomainKind.AUGMENTED)
    )

    assert physical.certificate.domain is DomainKind.PHYSICAL
    assert augmented.certificate.domain is DomainKind.AUGMENTED
