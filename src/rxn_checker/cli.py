"""Command-line selection, execution, and rendering."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from .checks.registry import CHECK_REGISTRY, PROFILES
from .checks.runner import run_checks
from .loading import load_case
from .reporting import render_json, render_text
from .results import Verdict


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated check list.")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rxn-checker",
        description="Check a reaction case with a selected dependency profile.",
    )
    parser.add_argument(
        "case",
        nargs="?",
        type=Path,
        help="case YAML file, or a directory containing case.yaml",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--checks", type=_csv, help="comma-separated checks only")
    parser.add_argument("--skip", type=_csv, default=(), help="checks to exclude")
    parser.add_argument("--format", choices=("text", "json"), dest="format")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-checks", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _case_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "case.yaml"
    return path.resolve()


def _loading_error(path: Path, error: Exception, output_format: str) -> str:
    message = str(error.args[0]) if len(error.args) == 1 else str(error)
    if output_format == "json":
        return json.dumps(
            {
                "case_name": None,
                "source": str(path),
                "overall": Verdict.ERROR.value,
                "error": f"{type(error).__name__}: {message}",
            },
            indent=2,
        ) + "\n"
    return (
        f"rxn-checker: ERROR\nSource: {path}\n\n"
        f"Case loading\n  ERROR    {type(error).__name__}: {message}\n\n"
        "Overall: ERROR\n"
    )


def _list_checks() -> str:
    return "\n".join(
        f"{spec.id:<32} {spec.stage.value:<10} {spec.name}"
        for spec in CHECK_REGISTRY
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.list_checks:
        sys.stdout.write(_list_checks())
        return 0
    if arguments.case is None:
        _parser().error("CASE is required unless --list-checks is used")

    case_path = _case_path(arguments.case)
    try:
        case = load_case(case_path)
        configured_exclude = tuple(case.check_config.get("exclude", ()))
        excluded = tuple(dict.fromkeys((*configured_exclude, *arguments.skip)))
        run = run_checks(
            case,
            profile=arguments.profile,
            exclude=excluded,
            only=arguments.checks,
            debug=arguments.debug,
        )
    except Exception as error:
        if arguments.debug:
            raise
        rendered = _loading_error(case_path, error, arguments.format or "text")
        sys.stdout.write(rendered)
        return 2

    output_format = arguments.format or str(case.report_config.get("format", "text"))
    profile = arguments.profile or str(case.check_config.get("profile", "physical"))
    if output_format == "json":
        rendered = render_json(run)
    else:
        rendered = render_text(
            run,
            profile=None if arguments.checks else profile,
            source=case_path,
            verbosity=str(case.report_config.get("verbosity", "failures")),
        )

    configured_output = case.report_config.get("output")
    output = arguments.output or (
        case_path.parent / str(configured_output) if configured_output else None
    )
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    if run.overall is Verdict.ERROR:
        return 2
    return 0 if run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
