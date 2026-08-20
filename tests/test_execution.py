"""Phase 3 result, DAG, selection, and rendering tests."""

import json
from pathlib import Path

import pytest

from rxn_checker import AnalysisContext, Verdict, load_case
from rxn_checker.checks import (
    CheckScope,
    CheckSpec,
    Stage,
    execute_plan,
    plan_checks,
    validate_registry,
)
from rxn_checker.cli import main
from rxn_checker.reporting import render_json, render_text
from rxn_checker.results import Finding, Role


ROOT = Path(__file__).parents[1]
PROFILES = frozenset(("physical",))


def _spec(
    check_id,
    run,
    *,
    stage=Stage.PHYSICAL,
    requires=(),
    role=Role.BLOCKING,
):
    return CheckSpec(
        check_id,
        check_id.replace("_", " ").title(),
        stage,
        CheckScope.CASE,
        requires,
        role is Role.BLOCKING,
        PROFILES,
        run,
        role,
    )


def _finding(verdict=Verdict.PASS):
    def run(context, dependencies):
        return Finding(context.case.name, verdict, verdict.value)

    return run


def test_dependency_plan_is_deterministic_and_deduplicated() -> None:
    registry = (
        _spec("root", _finding()),
        _spec("left", _finding(), requires=("root",)),
        _spec("right", _finding(), requires=("root",)),
        _spec("leaf", _finding(), requires=("left", "right")),
    )

    planned = plan_checks(profile=None, only=("leaf",), registry=registry)

    assert tuple(spec.id for spec in planned) == ("root", "left", "right", "leaf")


def test_explicit_dependency_conflict_is_rejected() -> None:
    registry = (
        _spec("root", _finding()),
        _spec("leaf", _finding(), requires=("root",)),
    )

    with pytest.raises(ValueError, match="require excluded"):
        plan_checks(
            profile=None,
            only=("leaf",),
            exclude=("root",),
            registry=registry,
        )


def test_registry_rejects_cycles_and_unknown_dependencies() -> None:
    cyclic = (
        _spec("one", _finding(), requires=("two",)),
        _spec("two", _finding(), requires=("one",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_registry(cyclic)
    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_registry((_spec("one", _finding(), requires=("missing",)),))


def test_duplicate_dependency_paths_do_not_rerun_checks() -> None:
    case = load_case(ROOT / "example_case")
    calls = {"root": 0, "left": 0, "right": 0}

    def counted(check_id):
        def run(context, dependencies):
            calls[check_id] += 1
            return Finding(context.case.name, Verdict.PASS, "done")

        return run

    registry = (
        _spec("root", counted("root")),
        _spec("left", counted("left"), requires=("root",)),
        _spec("right", counted("right"), requires=("root",)),
    )
    plan = plan_checks(profile=None, only=("left", "right"), registry=registry)

    run = execute_plan(case, plan)

    assert calls == {"root": 1, "left": 1, "right": 1}
    assert run.overall is Verdict.PASS


def test_chemistry_failure_finishes_gate_then_stops_later_stage() -> None:
    case = load_case(ROOT / "example_case")
    calls = []

    def record(check_id, verdict):
        def run(context, dependencies):
            calls.append(check_id)
            return Finding(context.case.name, verdict, check_id)

        return run

    plan = (
        _spec("chem_fail", record("chem_fail", Verdict.FAIL), stage=Stage.CHEMISTRY),
        _spec("chem_pass", record("chem_pass", Verdict.PASS), stage=Stage.CHEMISTRY),
        _spec("physical", record("physical", Verdict.PASS)),
    )

    result = execute_plan(case, plan)

    assert calls == ["chem_fail", "chem_pass"]
    assert result.results["physical"].verdict is Verdict.SKIPPED
    assert "chemistry stage" in result.results["physical"].findings[0].summary


def test_unknown_prerequisite_skips_dependent_but_error_is_distinct() -> None:
    case = load_case(ROOT / "example_case")

    def broken(context, dependencies):
        raise RuntimeError("implementation fault")

    plan = (
        _spec("unknown", _finding(Verdict.UNKNOWN)),
        _spec("dependent", _finding(), requires=("unknown",)),
        _spec("broken", broken),
    )
    result = execute_plan(case, plan, fail_fast="none")

    assert result.results["unknown"].verdict is Verdict.UNKNOWN
    assert result.results["dependent"].verdict is Verdict.SKIPPED
    assert result.results["broken"].verdict is Verdict.ERROR
    assert result.overall is Verdict.ERROR


def test_context_constructs_shared_analysis_objects_once(monkeypatch) -> None:
    case = load_case(ROOT / "example_case")
    context = AnalysisContext(case)
    network_calls = 0

    import rxn_checker.context as context_module

    real_network = context_module.build_network

    def counted_network(value):
        nonlocal network_calls
        network_calls += 1
        return real_network(value)

    monkeypatch.setattr(context_module, "build_network", counted_network)

    assert context.network is context.network
    assert context.expression_analyzer is context.expression_analyzer
    assert network_calls == 1


def test_text_and_json_render_the_same_structured_result() -> None:
    case = load_case(ROOT / "example_case")
    spec = _spec("sample", _finding())
    result = execute_plan(case, (spec,))

    text = render_text(result, registry=(spec,))
    payload = json.loads(render_json(result, registry=(spec,)))

    assert "rxn-checker: PASS" in text
    assert payload["overall"] == "PASS"
    assert payload["results"]["sample"]["verdict"] == "PASS"


def test_case_yaml_rejects_unknown_check_selection(tmp_path) -> None:
    source = (ROOT / "example_case" / "case.yaml").read_text(encoding="utf-8")
    source = source.replace("profile: robust", "profile: basic\n  include: [missing]")
    path = tmp_path / "case.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown checks"):
        load_case(path)


def test_cli_checks_selects_only_requested_check_and_prerequisites(
    capsys, tmp_path
) -> None:
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        (ROOT / "example_case" / "case.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    exit_code = main(
        (
            str(case_path),
            "--checks",
            "physical_lipschitz",
            "--format",
            "json",
        )
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 1
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == output
    assert payload["selected_checks"] == [
        "atom_conservation",
        "mass_conservation",
        "physical_rate_definedness",
        "physical_lipschitz",
    ]
    assert payload["results"]["physical_lipschitz"]["domain"] == "physical"
    assert payload["results"]["physical_lipschitz"]["prerequisites"] == {
        "physical_rate_definedness": "PASS"
    }
    certificate = payload["results"]["physical_lipschitz"]["findings"][0]["evidence"]
    assert certificate["kind"] == "lipschitz_certificate"
    assert certificate["data"]["domain"] == "physical"
    assert "constant_bound" in certificate["data"]
