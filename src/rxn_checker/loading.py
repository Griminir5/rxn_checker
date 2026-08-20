"""Strict schema-1 case loading with requested-family imports."""

from collections.abc import Mapping
import hashlib
from importlib import import_module, util
from pathlib import Path
import sys
from types import ModuleType

import yaml

from .case import Case
from .domain import (
    GAS_CONSTANT,
    ConcentrationModel,
    DomainSpec,
    Interval,
    TotalConstraint,
)
from .model import CaseSymbols, Phase, Reaction, Species, parse_rational
from .reactions import BUILTIN_FAMILIES
from .species import PROPERTY_REGISTRY, PropertyRegistry


_TOP_LEVEL_KEYS = {
    "schema",
    "species",
    "inerts",
    "reactions",
    "parameters",
    "domain",
    "checks",
    "report",
}


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _reject_unknown(config: Mapping, allowed: set[str], label: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {label} keys: " + ", ".join(sorted(map(str, unknown))) + "."
        )


def _string_list(
    config: Mapping,
    key: str,
    *,
    optional: bool = False,
) -> tuple[str, ...]:
    if key not in config:
        if optional:
            return ()
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    values = config[key]
    if not isinstance(values, list):
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    result = tuple(values)
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"Case '{key}' entries must be non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError(f"Case '{key}' entries must be unique.")
    return result


def _pair(config: Mapping, key: str) -> Interval:
    values = config.get(key)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"Case parameter '{key}' must contain [lower, upper].")
    interval = Interval(values[0], values[1])
    if interval.lower >= interval.upper:
        raise ValueError(f"Case parameter '{key}' bounds are not ordered.")
    return interval


def _load_parameters(
    value: object,
    symbols: CaseSymbols,
) -> dict:
    config = _mapping(value, "Case 'parameters'")
    _reject_unknown(config, {"temperature", "pressure"}, "parameter")
    if set(config) != {"temperature", "pressure"}:
        raise ValueError("Case parameters must define temperature and pressure.")
    return {
        symbols.temperature: _pair(config, "temperature"),
        symbols.pressure: _pair(config, "pressure"),
    }


def _resolved_bounds(
    value: object,
    species_ids: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    config = _mapping(value, f"Domain '{label}'")
    _reject_unknown(config, {"default", "overrides"}, f"domain {label}")
    if "default" not in config:
        raise ValueError(f"Domain '{label}' requires a default value.")
    default = parse_rational(config["default"], label=f"Default {label}")
    overrides = _mapping(config.get("overrides", {}), f"Domain '{label}.overrides'")
    unknown = set(overrides) - set(species_ids)
    if unknown:
        raise ValueError(
            f"Domain '{label}' references unknown species: "
            + ", ".join(sorted(unknown))
            + "."
        )
    return {
        species_id: parse_rational(
            overrides.get(species_id, default),
            label=f"{label} for '{species_id}'",
        )
        for species_id in species_ids
    }


def _load_total(
    value: object | None,
    *,
    phase: Phase,
    species: tuple[Species, ...],
    symbols: CaseSymbols,
    parameters: Mapping,
) -> TotalConstraint | None:
    label = phase.value
    if value is None:
        return None
    config = _mapping(value, f"Domain total '{label}'")
    _reject_unknown(config, {"mode", "species", "value"}, f"{label} total")
    mode = config.get("mode", "none")
    allowed = {"none", "explicit"}
    if phase is Phase.GAS:
        allowed.add("ideal_gas_minimum")
    if mode not in allowed:
        raise ValueError(
            f"{label.title()} total mode must be one of: "
            + ", ".join(sorted(allowed))
            + "."
        )

    selected_ids = tuple(item.id for item in species if item.phase is phase)
    if "species" in config:
        configured = config["species"]
        if not isinstance(configured, list):
            raise ValueError(f"Domain {label} total species must be a YAML sequence.")
        selected_ids = tuple(configured)
        if any(not isinstance(item, str) or not item for item in selected_ids):
            raise ValueError(f"Domain {label} total species must be non-empty strings.")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"Domain {label} total species must be unique.")

    by_id = {item.id: item for item in species}
    unknown = set(selected_ids) - set(by_id)
    wrong_phase = {
        species_id
        for species_id in selected_ids
        if species_id in by_id and by_id[species_id].phase is not phase
    }
    if unknown:
        raise ValueError(
            f"Domain {label} total references unknown species: "
            + ", ".join(sorted(unknown))
            + "."
        )
    if wrong_phase:
        raise ValueError(
            f"Domain {label} total includes species from another phase: "
            + ", ".join(sorted(wrong_phase))
            + "."
        )
    if mode != "none" and not selected_ids:
        raise ValueError(f"Domain {label} total requires at least one species.")
    if mode == "none" and ("species" in config or "value" in config):
        raise ValueError(f"Domain {label} total options require a non-none mode.")
    if mode == "none":
        return None
    if mode == "explicit":
        if "value" not in config:
            raise ValueError(f"Explicit {label} total requires a value.")
        minimum = parse_rational(config["value"], label=f"Explicit {label} total")
    else:
        if "value" in config:
            raise ValueError("ideal_gas_minimum does not accept an explicit value.")
        pressure = parameters[symbols.pressure]
        temperature = parameters[symbols.temperature]
        if pressure.lower <= 0:
            raise ValueError("Ideal-gas minimum requires positive pressure.")
        minimum = pressure.lower / (GAS_CONSTANT * temperature.upper)
    return TotalConstraint(
        label,
        tuple(symbols.concentration(item) for item in selected_ids),
        minimum,
    )


def _load_domain(
    value: object,
    species: tuple[Species, ...],
    symbols: CaseSymbols,
    parameters: Mapping,
) -> DomainSpec:
    config = _mapping(value, "Case 'domain'")
    _reject_unknown(
        config,
        {"concentration_model", "upper", "excursion_lower", "totals"},
        "domain",
    )
    try:
        model = ConcentrationModel(config.get("concentration_model"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Domain concentration_model must be 'independent' or 'chamfered'."
        ) from error

    species_ids = tuple(item.id for item in species)
    upper_by_id = _resolved_bounds(config.get("upper"), species_ids, "upper")
    excursion_by_id = _resolved_bounds(
        config.get("excursion_lower"), species_ids, "excursion_lower"
    )
    totals = _mapping(config.get("totals", {}), "Domain 'totals'")
    _reject_unknown(totals, {"gas", "solid"}, "domain totals")
    constraints = tuple(
        constraint
        for constraint in (
            _load_total(
                totals.get("gas"),
                phase=Phase.GAS,
                species=species,
                symbols=symbols,
                parameters=parameters,
            ),
            _load_total(
                totals.get("solid"),
                phase=Phase.SOLID,
                species=species,
                symbols=symbols,
                parameters=parameters,
            ),
        )
        if constraint is not None
    )
    return DomainSpec(
        symbols,
        model,
        {symbols.concentration(item): upper_by_id[item] for item in species_ids},
        {
            symbols.concentration(item): excursion_by_id[item]
            for item in species_ids
        },
        parameters,
        constraints,
    )


def _load_checks(value: object | None) -> dict[str, object]:
    if value is None:
        return {"profile": "physical", "include": (), "exclude": (), "fail_fast": "stage"}
    config = _mapping(value, "Case 'checks'")
    _reject_unknown(config, {"profile", "include", "exclude", "fail_fast"}, "checks")
    profile = config.get("profile", "physical")
    if profile not in {"basic", "physical", "robust", "analysis", "all"}:
        raise ValueError(f"Unknown check profile '{profile}'.")
    include = _string_list(config, "include", optional=True)
    exclude = _string_list(config, "exclude", optional=True)
    fail_fast = config.get("fail_fast", "stage")
    if fail_fast not in {"stage", "none"}:
        raise ValueError("Check fail_fast must be 'stage' or 'none'.")
    from .checks.registry import plan_checks

    plan_checks(
        profile=profile,
        include=include,
        exclude=exclude,
    )
    return {
        "profile": profile,
        "include": include,
        "exclude": exclude,
        "fail_fast": fail_fast,
    }


def _load_report(value: object | None) -> dict[str, object]:
    if value is None:
        return {"verbosity": "failures", "format": "text"}
    config = _mapping(value, "Case 'report'")
    _reject_unknown(config, {"verbosity", "format"}, "report")
    verbosity = config.get("verbosity", "failures")
    if verbosity not in {"summary", "failures", "full"}:
        raise ValueError("Report verbosity must be 'summary', 'failures', or 'full'.")
    output_format = config.get("format", "text")
    if output_format not in {"text", "json"}:
        raise ValueError("Report format must be 'text' or 'json'.")
    return {"verbosity": verbosity, "format": output_format}


def _selector(selector: str) -> tuple[str, str | None]:
    parts = selector.split(".")
    if len(parts) not in (1, 2) or any(not part.isidentifier() for part in parts):
        raise ValueError(
            f"Invalid reaction selector '{selector}'; expected "
            "'family' or 'family.reaction'."
        )
    return parts[0], parts[1] if len(parts) == 2 else None


def _local_family_module(path: Path, family_id: str) -> ModuleType:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    module_name = f"_rxn_checker_case_{digest}_{family_id}"
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local reaction family '{family_id}'.")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _family_module(case_directory: Path, family_id: str) -> ModuleType:
    local_path = case_directory / "reactions" / f"{family_id}.py"
    if local_path.is_file():
        return _local_family_module(local_path, family_id)
    try:
        module_name = BUILTIN_FAMILIES[family_id]
    except KeyError as error:
        raise ValueError(f"Unknown reaction family '{family_id}'.") from error
    return import_module(module_name)


def _build_family(
    case_directory: Path,
    family_id: str,
    symbols: CaseSymbols,
) -> Mapping[str, Reaction]:
    module = _family_module(case_directory, family_id)
    builder = getattr(module, "build_family", None)
    if not callable(builder):
        raise TypeError(
            f"Reaction family '{family_id}' must define callable build_family()."
        )
    built = builder(symbols)
    if not isinstance(built, Mapping) or not built:
        raise TypeError(
            f"Reaction family '{family_id}' must return a non-empty mapping."
        )

    reactions: dict[str, Reaction] = {}
    for local_id, reaction in built.items():
        if not isinstance(local_id, str) or not local_id.isidentifier():
            raise ValueError(
                f"Reaction family '{family_id}' returned invalid id '{local_id}'."
            )
        if not isinstance(reaction, Reaction):
            raise TypeError(
                f"Reaction family '{family_id}' returned a non-Reaction value."
            )
        expected_id = f"{family_id}.{local_id}"
        if reaction.id != expected_id:
            raise ValueError(
                f"Reaction family '{family_id}' key '{local_id}' returned "
                f"reaction id '{reaction.id}', expected '{expected_id}'."
            )
        reactions[local_id] = reaction
    return reactions


def _load_reactions(
    selectors: tuple[str, ...],
    symbols: CaseSymbols,
    case_directory: Path,
) -> tuple[Reaction, ...]:
    parsed = tuple(_selector(selector) for selector in selectors)
    families: dict[str, Mapping[str, Reaction]] = {}
    for family_id, _ in parsed:
        if family_id not in families:
            try:
                families[family_id] = _build_family(
                    case_directory, family_id, symbols
                )
            except Exception as error:
                if isinstance(error, (ValueError, TypeError)):
                    raise
                raise RuntimeError(
                    f"Could not build reaction family '{family_id}': {error}"
                ) from error

    selected: list[Reaction] = []
    selected_ids: set[str] = set()
    for family_id, local_id in parsed:
        family = families[family_id]
        local_ids = tuple(family) if local_id is None else (local_id,)
        for reaction_name in local_ids:
            if reaction_name not in family:
                available = ", ".join(family)
                raise ValueError(
                    f"Unknown reaction '{family_id}.{reaction_name}'. "
                    f"Available reactions: {available}."
                )
            reaction = family[reaction_name]
            if reaction.id in selected_ids:
                raise ValueError(
                    f"Reaction '{reaction.id}' was selected more than once."
                )
            selected_ids.add(reaction.id)
            selected.append(reaction)
    return tuple(selected)


def load_case(
    path: str | Path,
    *,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
) -> Case:
    """Load and validate one schema-1 case without importing unselected families."""

    path = Path(path)
    if path.is_dir():
        path = path / "case.yaml"
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    config = _mapping(loaded, "Case document")
    _reject_unknown(config, _TOP_LEVEL_KEYS, "top-level case")
    if config.get("schema") != 1 or isinstance(config.get("schema"), bool):
        raise ValueError("Case 'schema' must be integer 1.")

    species_ids = _string_list(config, "species")
    inert_species = _string_list(config, "inerts", optional=True)
    selectors = _string_list(config, "reactions")
    if not selectors:
        raise ValueError("Case must select at least one reaction.")

    missing_species = [
        species_id
        for species_id in species_ids
        if species_id not in property_registry.records
    ]
    if missing_species:
        raise ValueError("Unknown case species: " + ", ".join(missing_species) + ".")
    species = tuple(property_registry.get_record(item) for item in species_ids)
    symbols = CaseSymbols.for_species(species_ids)
    parameters = _load_parameters(config.get("parameters"), symbols)
    domain_spec = _load_domain(
        config.get("domain"), species, symbols, parameters
    )
    reactions = _load_reactions(selectors, symbols, path.parent)

    return Case(
        name=path.parent.name,
        species=species,
        symbols=symbols,
        reactions=reactions,
        domain_spec=domain_spec,
        inert_species=inert_species,
        check_config=_load_checks(config.get("checks")),
        report_config=_load_report(config.get("report")),
    )


__all__ = ("load_case",)
