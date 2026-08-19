import unittest

from rxn_checker import Case, Reaction, StateVariables, VariableBounds
from rxn_checker.checks import CheckContext, CheckStatus
from rxn_checker.checks.equilibria import CHECK as EQUILIBRIA_CHECK
from rxn_checker.reporting import build_check_report
from rxn_checker.species import PropertyRegistry, SpeciesProperties
from tests import make_state_bounds


class CheckReportTests(unittest.TestCase):
    def test_equilibrium_relationship_is_summarized_in_reading_order(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        cee = states.concentration("C")
        reaction = Reaction(
            name="catalysed",
            family="example",
            reactants={"A": 1},
            products={"B": 1},
            catalysts=("C",),
            rate=aye * cee,
        )
        case = Case(
            "readable",
            states,
            (reaction,),
            make_state_bounds(states),
        )

        report = build_check_report(case, checks=(EQUILIBRIA_CHECK,))

        self.assertIn("Equilibrium relationships [equilibria; case]", report.text)
        self.assertLess(
            report.text.index("Branch 1: A = 0."),
            report.text.index("Branch 2: C = 0."),
        )
        self.assertIn(
            "The complete physical steady-state relationship has 2 branches.",
            report.text,
        )
        self.assertNotIn("Source operations", report.text)

    def test_failed_checks_include_atom_and_mass_diagnostics(self) -> None:
        states = StateVariables(("CH4", "CO2"))
        reaction = Reaction(
            name="unbalanced",
            family="methane",
            reactants={"CH4": 1},
            products={"CO2": 1},
            rate=states.concentration("CH4"),
        )
        bounds = make_state_bounds(states)
        bounds[states.concentration("CO2")] = VariableBounds(0, 1000, 0)
        case = Case(
            name="bad-case",
            states=states,
            reactions=(reaction,),
            state_bounds=bounds,
        )

        report = build_check_report(case, source="bad-case/case.yaml")

        self.assertFalse(report.passed)
        self.assertEqual(report.overall_status, CheckStatus.FAIL)
        self.assertEqual(report.status_counts[CheckStatus.FAIL], 2)
        self.assertIn("Atom conservation [atom_conservation; reaction]", report.text)
        self.assertIn("methane.unbalanced: FAIL", report.text)
        self.assertIn("H imbalance: -4", report.text)
        self.assertIn("O imbalance: 2", report.text)
        self.assertIn("Mass conservation [mass_conservation; reaction]", report.text)
        self.assertIn("Overall: FAIL", report.text)

    def test_unavailable_mass_check_has_its_own_status(self) -> None:
        registry = PropertyRegistry(
            {
                "A": SpeciesProperties("A", "gas", {"X": 1}),
                "B": SpeciesProperties("B", "gas", {"X": 1}, 1.0),
            }
        )
        states = StateVariables(("A", "B"))
        reaction = Reaction(
            name="conversion",
            family="example",
            reactants={"A": 1},
            products={"B": 1},
            rate=states.concentration("A"),
        )
        bounds = make_state_bounds(states)
        bounds[states.concentration("B")] = VariableBounds(0, 1000, 0)
        case = Case(
            name="missing-data",
            states=states,
            reactions=(reaction,),
            state_bounds=bounds,
        )

        report = build_check_report(
            case,
            context=CheckContext(property_registry=registry),
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.overall_status, CheckStatus.UNAVAILABLE)
        self.assertEqual(report.status_counts[CheckStatus.PASS], 8)
        self.assertEqual(report.status_counts[CheckStatus.UNAVAILABLE], 1)
        self.assertIn("example.conversion: PASS", report.text)
        self.assertIn("example.conversion: UNAVAILABLE", report.text)
        self.assertIn("molecular weight is missing for species: A", report.text)
