"""Mathematical regression tests for the Phase 4 expression analyzer."""

from collections import Counter
from pathlib import Path

import sympy as sp

from rxn_checker import AnalysisContext, load_case
from rxn_checker.domain import ConcentrationDomain, DomainKind, Interval
from rxn_checker.proof import (
    ExpressionAnalyzer,
    ProofVerdict,
    Sign,
    SignRequirement,
)


ROOT = Path(__file__).parents[1]


def _domain(*, c_bounds=(0, 2), d_bounds=(-1, 3)):
    c, d = sp.symbols("c d", real=True)
    domain = ConcentrationDomain(
        DomainKind.PHYSICAL,
        {c: Interval(*c_bounds), d: Interval(*d_bounds)},
        {},
    )
    return c, d, domain


def test_arithmetic_bounds_are_sound_and_affine_bounds_are_exact() -> None:
    c, d, domain = _domain()
    analyzer = ExpressionAnalyzer()

    affine = analyzer.bounds(2 * c - d + 1, domain)
    product = analyzer.bounds(c * d, domain)
    square = analyzer.bounds(c**2, domain)

    assert (affine.lower, affine.upper, affine.exact) == (-2, 6, True)
    assert affine.lower_witness[c] == 0
    assert affine.lower_witness[d] == 3
    assert (product.lower, product.upper) == (-2, 6)
    assert (square.lower, square.upper) == (0, 4)


def test_generic_function_bounds_cover_the_supported_rules() -> None:
    c, d, domain = _domain()
    analyzer = ExpressionAnalyzer()

    expected = {
        sp.Abs(d): (0, 3),
        sp.Min(c, d): (-1, 2),
        sp.Max(c, d): (0, 3),
        sp.exp(c): (1, sp.exp(2)),
        sp.log(c + 1): (0, sp.log(3)),
        sp.sin(c): (-1, 1),
        sp.cos(c): (-1, 1),
        sp.sinh(c): (0, sp.sinh(2)),
        sp.cosh(d): (1, sp.cosh(3)),
        sp.tanh(c): (0, sp.tanh(2)),
        sp.atan(c): (0, sp.atan(2)),
    }

    for expression, endpoints in expected.items():
        result = analyzer.bounds(expression, domain)
        assert (result.lower, result.upper) == endpoints


def test_rational_expression_bounds_require_a_separated_denominator() -> None:
    c, d, domain = _domain()
    analyzer = ExpressionAnalyzer()
    expression = (c + 1) / (d + 2)

    bounds = analyzer.bounds(expression, domain)
    defined = analyzer.defined(expression, domain)

    assert (bounds.lower, bounds.upper) == (sp.Rational(1, 5), 3)
    assert defined.verdict is ProofVerdict.PASS


def test_definedness_guards_report_exact_failures_and_unknown_functions() -> None:
    c, _, domain = _domain()
    analyzer = ExpressionAnalyzer()

    reciprocal = analyzer.defined(1 / c, domain)
    logarithm = analyzer.defined(sp.log(c), domain)
    square_root = analyzer.defined(sp.sqrt(c), domain)
    discontinuous = analyzer.defined(sp.floor(c), domain)

    assert reciprocal.verdict is ProofVerdict.FAIL
    assert reciprocal.requirement is SignRequirement.NONZERO
    assert reciprocal.witness[c] == 0
    assert logarithm.verdict is ProofVerdict.FAIL
    assert logarithm.decisive_subexpression == c
    assert square_root.verdict is ProofVerdict.PASS
    assert discontinuous.verdict is ProofVerdict.UNKNOWN
    assert discontinuous.decisive_subexpression == sp.floor(c)


def test_trigonometric_poles_report_the_guard_denominator() -> None:
    c = sp.Symbol("c", real=True)
    domain = ConcentrationDomain(
        DomainKind.PHYSICAL,
        {c: Interval(0, sp.pi)},
        {},
    )

    result = ExpressionAnalyzer().defined(sp.tan(c), domain)

    assert result.verdict is ProofVerdict.FAIL
    assert result.decisive_subexpression == sp.cos(c)
    assert result.witness[c] == sp.pi / 2


def test_positive_and_open_positive_intervals_certify_reciprocals() -> None:
    c = sp.Symbol("c", real=True)
    positive = ConcentrationDomain(
        DomainKind.PHYSICAL,
        {c: Interval(1, 2)},
        {},
    )
    open_positive = ConcentrationDomain(
        DomainKind.PHYSICAL,
        {c: Interval(0, 2, lower_closed=False)},
        {},
    )
    analyzer = ExpressionAnalyzer()

    assert analyzer.defined(1 / c, positive).verdict is ProofVerdict.PASS
    assert analyzer.bounds(1 / c, positive).lower == sp.Rational(1, 2)
    assert analyzer.defined(1 / c, open_positive).verdict is ProofVerdict.PASS
    assert analyzer.sign(c, open_positive).sign is Sign.POSITIVE


def test_chamfered_total_proves_a_reciprocal_denominator_positive() -> None:
    case = load_case(ROOT / "reforming_case")
    domain = AnalysisContext(case).physical_domain
    gases = tuple(
        item.id for item in case.species if item.phase.value == "gas"
    )
    total = sum(case.symbols.concentration(item) for item in gases)
    analyzer = ExpressionAnalyzer()

    bounds = analyzer.bounds(total, domain)

    assert bounds.exact
    assert bounds.lower > 0
    assert analyzer.defined(1 / total, domain).verdict is ProofVerdict.PASS


def test_sign_proofs_use_bounds_and_bounded_exact_witness_search() -> None:
    c, _, domain = _domain()
    analyzer = ExpressionAnalyzer()

    square = analyzer.prove_sign(c**2, domain, SignRequirement.NONNEGATIVE)
    affine_failure = analyzer.prove_sign(c - 1, domain, SignRequirement.NONNEGATIVE)
    nonlinear_failure = analyzer.prove_sign(
        c * (1 - c),
        domain,
        SignRequirement.NONNEGATIVE,
    )

    assert square.verdict is ProofVerdict.PASS
    assert affine_failure.verdict is ProofVerdict.FAIL
    assert affine_failure.witness_value == -1
    assert nonlinear_failure.verdict is ProofVerdict.FAIL
    assert nonlinear_failure.witness[c] == 2
    assert nonlinear_failure.witness_value == -2


def test_loose_bounds_return_unknown_without_a_counterexample() -> None:
    c, _, domain = _domain(c_bounds=(0, 1))
    analyzer = ExpressionAnalyzer()

    proof = analyzer.prove_sign(sp.sin(c), domain, SignRequirement.NONNEGATIVE)

    assert proof.verdict is ProofVerdict.UNKNOWN
    assert proof.result.interval == analyzer.bounds(sp.sin(c), domain)


def test_analysis_is_cached_by_expression_domain_and_active_variables() -> None:
    c, d, domain = _domain()

    class CountingAnalyzer(ExpressionAnalyzer):
        def __init__(self):
            super().__init__()
            self.calls = Counter()

        def _compute_bounds(self, expression, selected_domain, active):
            self.calls[expression] += 1
            return super()._compute_bounds(expression, selected_domain, active)

    analyzer = CountingAnalyzer()
    shared = c + 1
    expression = shared**2 + shared

    first = analyzer.bounds(expression, domain, (c,))
    second = analyzer.bounds(expression, domain, (c,))
    shared_result = analyzer.bounds(shared, domain, (c,))

    assert first is second
    assert shared_result is analyzer.bounds(shared, domain, (c,))
    assert analyzer.calls[shared] == 1
    assert analyzer.bounds(expression, domain, (d,)) is not first
