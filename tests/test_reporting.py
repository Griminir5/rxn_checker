import unittest

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import CheckContext, CheckStatus
from rxn_checker.reporting import build_check_report
from rxn_checker.species import PropertyRegistry, SpeciesProperties


class CheckReportTests(unittest.TestCase):
    def test_failed_checks_include_atom_and_mass_diagnostics(self) -> None:
        states = StateVariables(("CH4", "CO2"))
        reaction = Reaction(
            name="unbalanced",
            family="methane",
            reactants={"CH4": 1},
            products={"CO2": 1},
            rate=states.concentration("CH4"),
        )
        case = Case(
            name="bad-case",
            states=states,
            reactions=(reaction,),
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
                "A": SpeciesProperties("A", {"X": 1}),
                "B": SpeciesProperties("B", {"X": 1}, 1.0),
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
        case = Case(
            name="missing-data",
            states=states,
            reactions=(reaction,),
        )

        report = build_check_report(
            case,
            context=CheckContext(property_registry=registry),
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.overall_status, CheckStatus.UNAVAILABLE)
        self.assertEqual(report.status_counts[CheckStatus.PASS], 2)
        self.assertEqual(report.status_counts[CheckStatus.UNAVAILABLE], 1)
        self.assertIn("example.conversion: PASS", report.text)
        self.assertIn("example.conversion: UNAVAILABLE", report.text)
        self.assertIn("molecular weight is missing for species: A", report.text)
