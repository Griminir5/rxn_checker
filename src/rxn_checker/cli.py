"""Command-line interface for rxn-checker."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .loading import load_case
from .reporting import build_check_report


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run checks for one case and return a process exit code."""

    arguments = _parser().parse_args(argv)
    case_path = _case_path(arguments.case)

    try:
        case = load_case(case_path)
    except Exception as error:
        report_text = _loading_error_report(case_path, error)
        sys.stdout.write(report_text)
        return 2

    report = build_check_report(case, source=case_path)
    sys.stdout.write(report.text)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
