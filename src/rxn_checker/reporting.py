"""Text and JSON renderers for structured run results."""

from collections.abc import Iterable, Mapping
from enum import Enum
import json
from pathlib import Path

import sympy as sp

from .checks.definitions import CheckScope, CheckSpec, Stage
from .checks.registry import CHECK_REGISTRY
from .results import Finding, Role, RunResult, Verdict


_STAGE_NAMES = {
    Stage.CHEMISTRY: "Chemistry",
    Stage.PHYSICAL: "Physical domain",
    Stage.AUGMENTED: "Augmented domain",
    Stage.ANALYSIS: "Analyses",
}


def _specs(registry: Iterable[CheckSpec]) -> dict[str, CheckSpec]:
    return {spec.id: spec for spec in registry}


def render_text(
    run: RunResult,
    *,
    profile: str | None = None,
    source: str | Path | None = None,
    verbosity: str = "failures",
    registry: Iterable[CheckSpec] = CHECK_REGISTRY,
) -> str:
    """Render concise human output without changing result semantics."""

    if verbosity not in {"summary", "failures", "full"}:
        raise ValueError("Unknown report verbosity.")
    by_id = _specs(registry)
    lines = [f"rxn-checker: {run.overall.value}", f"Case: {run.case_name}"]
    if profile is not None:
        lines.append(f"Profile: {profile}")
    if source is not None:
        lines.append(f"Source: {Path(source)}")

    if verbosity != "summary":
        current_stage: Stage | None = None
        for check_id in run.selected_checks:
            spec = by_id[check_id]
            result = run.results[check_id]
            if spec.stage is not current_stage:
                lines.extend(("", _STAGE_NAMES[spec.stage]))
                current_stage = spec.stage

            findings = result.findings
            if spec.scope is CheckScope.REACTION:
                reaction_findings = tuple(
                    item for item in findings if item.subject != run.case_name
                )
                case_findings = tuple(
                    item for item in findings if item.subject == run.case_name
                )
            else:
                reaction_findings = ()
                case_findings = findings
            passed = sum(item.verdict is Verdict.PASS for item in reaction_findings)
            show_passing_case_detail = False
            aggregate = (
                reaction_findings
                and passed == len(reaction_findings)
                and all(item.verdict is Verdict.PASS for item in case_findings)
                and (len(reaction_findings) > 1 or bool(case_findings))
            )
            if aggregate:
                lines.append(
                    f"  PASS     {spec.name:<34} "
                    f"{passed}/{len(reaction_findings)} reactions"
                )
                if verbosity == "failures" and case_findings:
                    findings = case_findings
                    show_passing_case_detail = True
                elif verbosity != "full":
                    continue
            else:
                progress = (
                    f" {passed}/{len(reaction_findings)} reactions"
                    if len(reaction_findings) > 1
                    else ""
                )
                name = f"{spec.name:<34}" if progress else spec.name
                lines.append(f"  {result.verdict.value:<8} {name}{progress}")

            visible = findings
            if verbosity == "failures" and result.verdict is not Verdict.PASS:
                visible = tuple(
                    item
                    for item in findings
                    if item.verdict is not Verdict.PASS
                    or (
                        item.evidence is not None
                        and item.evidence.kind == "negative_side_summary"
                    )
                )
            elif verbosity == "failures" and not show_passing_case_detail:
                visible = (
                    findings
                    if result.role is Role.ANALYSIS
                    else tuple(
                        item
                        for item in findings
                        if item.evidence is not None
                        and item.evidence.kind == "negative_side_summary"
                    )
                )
            for finding in visible:
                subject = (
                    ""
                    if finding.subject == run.case_name
                    else f"{finding.subject}: "
                )
                lines.append(
                    f"           {finding.verdict.value:<8} {subject}{finding.summary}"
                )

    lines.extend(("", f"Overall: {run.overall.value}"))
    return "\n".join(lines) + "\n"


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, sp.Basic):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finding_json(finding: Finding) -> dict[str, object]:
    rendered: dict[str, object] = {
        "subject": finding.subject,
        "verdict": finding.verdict.value,
        "summary": finding.summary,
    }
    if finding.evidence is not None:
        rendered["evidence"] = {
            "kind": finding.evidence.kind,
            "data": _json_value(finding.evidence.data),
        }
    return rendered


def render_json(
    run: RunResult,
    *,
    registry: Iterable[CheckSpec] = CHECK_REGISTRY,
) -> str:
    """Serialize every structured result, including evidence and timing."""

    by_id = _specs(registry)
    results = {}
    for check_id in run.selected_checks:
        spec = by_id[check_id]
        result = run.results[check_id]
        results[check_id] = {
            "name": spec.name,
            "stage": spec.stage.value,
            "domain": (
                spec.stage.value
                if spec.stage in {Stage.PHYSICAL, Stage.AUGMENTED}
                else None
            ),
            "scope": spec.scope.value,
            "role": result.role.value,
            "verdict": result.verdict.value,
            "requires": list(spec.requires),
            "prerequisites": {
                required: run.results[required].verdict.value
                for required in spec.requires
            },
            "duration_seconds": result.duration_seconds,
            "findings": [_finding_json(item) for item in result.findings],
        }
    payload = {
        "case_name": run.case_name,
        "selected_checks": list(run.selected_checks),
        "overall": run.overall.value,
        "results": results,
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


__all__ = ("render_json", "render_text")
