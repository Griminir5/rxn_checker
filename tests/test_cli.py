from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rxn_checker.cli import REPORT_FILENAME, main


class CliTests(unittest.TestCase):
    def _write_case(self, directory: Path, species: str = "Aye") -> Path:
        path = directory / "case.yaml"
        path.write_text(
            f"species:\n  - {species}\n  - Bee\n" "reactions:\n  - aye_to_bee.simple\n",
            encoding="utf-8",
        )
        return path

    def _run(self, argument: Path) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main((str(argument),))
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_case_file_prints_and_writes_a_passing_report(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            case_path = self._write_case(directory)

            exit_code, stdout, stderr = self._run(case_path)

            report_path = directory / REPORT_FILENAME
            report_text = report_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            self.assertTrue(stdout.startswith(report_text))
            self.assertIn("aye_to_bee.simple", report_text)
            self.assertIn(
                "Atom conservation [atom_conservation; reaction]", report_text
            )
            self.assertIn(
                "Mass conservation [mass_conservation; reaction]", report_text
            )
            self.assertEqual(report_text.count("aye_to_bee.simple: PASS"), 2)
            self.assertIn("Case loading: PASS", report_text)
            self.assertIn("Overall: PASS", report_text)
            self.assertIn(f"Report written to {report_path}", stdout)

    def test_failed_check_report_returns_exit_one(self) -> None:
        failed_report = SimpleNamespace(
            text="checks failed\n",
            passed=False,
        )
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            case_path = self._write_case(directory)

            with patch(
                "rxn_checker.cli.build_check_report",
                return_value=failed_report,
            ):
                exit_code, stdout, stderr = self._run(case_path)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stderr, "")
            self.assertTrue(stdout.startswith(failed_report.text))
            self.assertEqual(
                (directory / REPORT_FILENAME).read_text(encoding="utf-8"),
                failed_report.text,
            )

    def test_case_directory_resolves_case_yaml(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            self._write_case(directory)

            exit_code, stdout, _ = self._run(directory)

            self.assertEqual(exit_code, 0)
            self.assertIn(f"Source: {directory / 'case.yaml'}", stdout)
            self.assertTrue((directory / REPORT_FILENAME).is_file())

    def test_loading_error_is_printed_written_and_returns_exit_two(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            case_path = self._write_case(directory, species="Missing")

            exit_code, stdout, stderr = self._run(case_path)

            report_text = (directory / REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertEqual(exit_code, 2)
            self.assertEqual(stderr, "")
            self.assertTrue(stdout.startswith(report_text))
            self.assertIn("Case loading: ERROR", report_text)
            self.assertIn("Unknown case species: Missing", report_text)
            self.assertIn("Overall: ERROR", report_text)


if __name__ == "__main__":
    unittest.main()
