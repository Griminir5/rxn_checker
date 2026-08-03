"""Command-line interface for rxn-checker."""

from __future__ import annotations

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
    if len(error.args) == 1 and isinstance(error.args[0], str):
        message = error.args[0]
    else:
        message = str(error)
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


def _write_report(case_path: Path, text: str) -> Path:
    report_path = case_path.parent / REPORT_FILENAME
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run checks for one case and return a process exit code."""

    arguments = _parser().parse_args(argv)
    case_path = _case_path(arguments.case)

    try:
        case = load_case(case_path)
    except Exception as error:
        report_text = _loading_error_report(case_path, error)
        sys.stdout.write(report_text)
        try:
            report_path = _write_report(case_path, report_text)
        except OSError as write_error:
            print(f"Could not write report: {write_error}", file=sys.stderr)
        else:
            print(f"Report written to {report_path}")
        return 2

    report = build_check_report(case, source=case_path)
    sys.stdout.write(report.text)
    try:
        report_path = _write_report(case_path, report.text)
    except OSError as error:
        print(f"Could not write report: {error}", file=sys.stderr)
        return 2

    print(f"Report written to {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("main",)
