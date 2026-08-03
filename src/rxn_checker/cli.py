"""Command-line interface for rxn-checker."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .loading import load_case
from .reporting import build_check_report

REPORT_FILENAME = "rxn-checker-report.txt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rxn-checker",
        description="Run registered checks for a reaction case.",
    )
    parser.add_argument(
        "case",
        type=Path,
        help="case YAML file, or a directory containing case.yaml",
    )
    return parser


def _case_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "case.yaml"
    return path.resolve()


def _loading_error_report(case_path: Path, error: Exception) -> str:
    message = str(error.args[0]) if len(error.args) == 1 else str(error)
    return "\n".join(
        (
            "rxn-checker report",
            f"Source: {case_path}",
            "",
            "Case loading: ERROR",
            f"  {message}",
            "",
            "Summary",
            "  Overall: ERROR",
            "",
        )
    )


def _output_report(case_path: Path, text: str) -> bool:
    sys.stdout.write(text)
    report_path = case_path.parent / REPORT_FILENAME
    try:
        report_path.write_text(text, encoding="utf-8")
    except OSError as error:
        print(f"Could not write report: {error}", file=sys.stderr)
        return False
    print(f"Report written to {report_path}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run checks for one case and return a process exit code."""

    arguments = _parser().parse_args(argv)
    case_path = _case_path(arguments.case)

    try:
        case = load_case(case_path)
    except Exception as error:
        report_text = _loading_error_report(case_path, error)
        _output_report(case_path, report_text)
        return 2

    report = build_check_report(case, source=case_path)
    if not _output_report(case_path, report.text):
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
