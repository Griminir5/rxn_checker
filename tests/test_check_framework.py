import unittest

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import (
    CHECK_REGISTRY,
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
    aggregate_status,
    run_checks,
)
from rxn_checker.reporting import build_check_report
from tests import make_state_bounds


class CheckFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        states = StateVariables(("Aye", "Bee"))
        reaction = Reaction(
            name="conversion",
            family="example",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=states.concentration("Aye"),
        )
        self.case = Case(
            name="example",
            states=states,
            reactions=(reaction,),
            state_bounds=make_state_bounds(states),
        )

    def definition(self, check_id, run, scope=CheckScope.CASE):
        return CheckDefinition(
            id=check_id,
            name=check_id.replace("_", " ").title(),
            group="Test checks",
            scope=scope,
            run=run,
        )

    def test_registered_checks_are_explicit_and_ordered(self) -> None:
        self.assertEqual(
            tuple(check.id for check in CHECK_REGISTRY),
            (
                "atom_conservation",
                "mass_conservation",
                "rate_nonnegativity",
                "zero_at_depletion",
                "nonphysical_recovery",
                "stoichiometric_conservation",
                "equilibria",
                "terminal_faces",
            ),
        )
        self.assertEqual(
            tuple(execution.definition.id for execution in run_checks(self.case)),
            (
                "atom_conservation",
                "mass_conservation",
                "rate_nonnegativity",
                "zero_at_depletion",
                "nonphysical_recovery",
                "stoichiometric_conservation",
                "equilibria",
                "terminal_faces",
            ),
        )

    def test_case_check_can_return_sampled_status_and_numerical_values(self) -> None:
        def sampled(case, context):
            self.assertIs(case, self.case)
            self.assertIsInstance(context, CheckContext)
            return CheckOutcome(
                status=CheckStatus.SAMPLED_PASS,
                details=("No violations in sampled points.",),
                values=(
                    CheckValue("Samples", 128),
                    CheckValue("Minimum rate", 1.25e-6, "mol/m^3/s"),
                ),
            )

        report = build_check_report(
            self.case,
            checks=(self.definition("sampled_rate", sampled),),
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.overall_status, CheckStatus.SAMPLED_PASS)
        self.assertEqual(report.value_count, 2)
        self.assertIn("Case: SAMPLED_PASS", report.text)
        self.assertIn("Samples: 128", report.text)
        self.assertIn("Minimum rate: 1.25e-06 mol/m^3/s", report.text)

    def test_numerical_only_check_does_not_invent_a_status(self) -> None:
        def metric(case, context):
            return CheckOutcome(values=(CheckValue("Maximum rate", 42.0),))

        report = build_check_report(
            self.case,
            checks=(self.definition("maximum_rate", metric),),
        )

        self.assertIsNone(report.overall_status)
        self.assertTrue(report.passed)
        self.assertIn("Maximum rate: 42", report.text)
        self.assertIn("Overall: NO_STATUS", report.text)

    def test_unexpected_failure_is_indeterminate_and_does_not_stop_later_checks(
        self,
    ) -> None:
        calls = []

        def broken(case, context):
            calls.append("broken")
            raise RuntimeError("solver stopped")

        def following(case, context):
            calls.append("following")
            return CheckOutcome(status=CheckStatus.PASS)

        executions = run_checks(
            self.case,
            (
                self.definition("broken", broken),
                self.definition("following", following),
            ),
        )

        self.assertEqual(calls, ["broken", "following"])
        self.assertEqual(
            executions[0].outcomes[0].status,
            CheckStatus.INDETERMINATE,
        )
        self.assertIn("RuntimeError: solver stopped", executions[0].outcomes[0].details)
        self.assertEqual(executions[1].outcomes[0].status, CheckStatus.PASS)

    def test_status_aggregation_uses_documented_severity(self) -> None:
        outcomes = tuple(
            CheckOutcome(status=status)
            for status in (
                CheckStatus.PASS,
                CheckStatus.SAMPLED_PASS,
                CheckStatus.UNAVAILABLE,
                CheckStatus.INDETERMINATE,
                CheckStatus.FAIL,
            )
        )
        self.assertEqual(aggregate_status(outcomes), CheckStatus.FAIL)

    def test_duplicate_check_ids_are_rejected(self) -> None:
        def passing(case, context):
            return CheckOutcome(status=CheckStatus.PASS)

        check = self.definition("duplicate", passing)
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            run_checks(self.case, (check, check))
