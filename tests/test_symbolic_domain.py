import unittest

import sympy as sp

from rxn_checker.checks.symbolic_domain import LinearDomain


class LinearDomainTests(unittest.TestCase):
    def test_free_variable_simplex_candidate_is_exactly_validated(self) -> None:
        x = sp.Symbol("x", real=True)
        domain = LinearDomain(
            (x,),
            (
                sp.Ge(x, 2, evaluate=False),
                sp.Le(x, 0, evaluate=False),
            ),
            (x,),
        )

        self.assertEqual(domain.feasible(), (False, None))

    def test_shifted_finite_box_finds_a_strict_exact_witness(self) -> None:
        x, y, z = sp.symbols("x y z", real=True)
        weak = (
            sp.Ge(x, 0, evaluate=False),
            sp.Le(x, 2, evaluate=False),
            sp.Ge(y, -2, evaluate=False),
            sp.Le(y, 2, evaluate=False),
            sp.Ge(z, 0, evaluate=False),
            sp.Le(z, 0, evaluate=False),
            sp.Ge(-2 * x - 2 * y - 2 * z - 3, 0, evaluate=False),
            sp.Ge(-x - y - z - 2, 0, evaluate=False),
            sp.Ge(-x + y + 3, 0, evaluate=False),
        )
        strict = (x - 2 * y - z + 1, y - 2 * z + 3)
        domain = LinearDomain(
            (x, y, z),
            weak,
            strict,
            (
                (sp.S.Zero, sp.Integer(2)),
                (-sp.Integer(2), sp.Integer(2)),
                (sp.S.Zero, sp.S.Zero),
            ),
        )

        feasible, point = domain.feasible()

        self.assertTrue(feasible)
        self.assertIsNotNone(point)
        self.assertTrue(all(bool(constraint.subs(point)) for constraint in weak))
        self.assertTrue(all(expression.subs(point) > 0 for expression in strict))


if __name__ == "__main__":
    unittest.main()
