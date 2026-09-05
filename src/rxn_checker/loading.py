"""Strict schema-1 YAML loading and requested-family imports."""

import hashlib
import sys
from collections.abc import Mapping
from importlib import import_module, util
from pathlib import Path

import yaml

from .case import Case
from .domain import GAS_CONSTANT, ConcentrationModel, DomainSpec, Interval, TotalConstraint
from .model import CaseSymbols, Phase, Reaction, parse_rational
from .reactions import BUILTIN_FAMILIES
from .species import PROPERTY_REGISTRY, PropertyRegistry

_TOP_KEYS = {"schema", "species", "inerts", "reactions", "parameters", "domain", "checks", "report"}


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _keys(config, allowed, label):
    unknown = set(config) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} keys: " + ", ".join(sorted(map(str, unknown))) + ".")


def _strings(config, key, optional=False):
    if key not in config and optional:
        return ()
    values = config.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Case '{key}' must be a YAML sequence.")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
        raise ValueError(f"Case '{key}' entries must be non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"Case '{key}' entries must be unique.")
    return tuple(values)


def _parameters(value, symbols):
    config = _mapping(value, "Case 'parameters'")
    _keys(config, ("temperature", "pressure"), "parameter")
    if set(config) != {"temperature", "pressure"}:
        raise ValueError("Case parameters must define temperature and pressure.")
    result = {}
    for key, symbol in (("temperature", symbols.temperature), ("pressure", symbols.pressure)):
        pair = config[key]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Case parameter '{key}' must contain [lower, upper].")
        interval = Interval(*pair)
        if interval.lower >= interval.upper:
            raise ValueError(f"Case parameter '{key}' bounds are not ordered.")
        result[symbol] = interval
    return result


def _bounds(value, species_ids, label):
    config = _mapping(value, f"Domain '{label}'")
    _keys(config, ("default", "overrides"), f"domain {label}")
    if "default" not in config:
        raise ValueError(f"Domain '{label}' requires a default value.")
    default = parse_rational(config["default"], label=f"Default {label}")
    overrides = _mapping(config.get("overrides", {}), f"Domain '{label}.overrides'")
    unknown = set(overrides) - set(species_ids)
    if unknown:
        raise ValueError(
            f"Domain '{label}' references unknown species: " + ", ".join(sorted(unknown)) + "."
        )
    return {
        item: parse_rational(overrides.get(item, default), label=f"{label} for '{item}'")
        for item in species_ids
    }


def _total(value, phase, species, symbols, parameters):
    if value is None:
        return None
    label = phase.value
    config = _mapping(value, f"Domain total '{label}'")
    _keys(config, ("mode", "species", "value"), f"{label} total")
    mode = config.get("mode", "none")
    allowed = {"none", "explicit"} | ({"ideal_gas_minimum"} if phase is Phase.GAS else set())
    if mode not in allowed:
        raise ValueError(
            f"{label.title()} total mode must be one of: " + ", ".join(sorted(allowed)) + "."
        )
    selected = tuple(item.id for item in species if item.phase is phase)
    if "species" in config:
        selected = tuple(config["species"] if isinstance(config["species"], list) else ())
        if not selected or any(not isinstance(item, str) or not item for item in selected):
            raise ValueError(f"Domain {label} total species must be non-empty strings.")
        if len(selected) != len(set(selected)):
            raise ValueError(f"Domain {label} total species must be unique.")
    by_id = {item.id: item for item in species}
    unknown = set(selected) - set(by_id)
    wrong = {item for item in selected if item in by_id and by_id[item].phase is not phase}
    if unknown or wrong:
        kind, values = (
            ("unknown species", unknown) if unknown else ("species from another phase", wrong)
        )
        raise ValueError(
            f"Domain {label} total includes {kind}: " + ", ".join(sorted(values)) + "."
        )
    if mode == "none":
        if "species" in config or "value" in config:
            raise ValueError(f"Domain {label} total options require a non-none mode.")
        return None
    if not selected:
        raise ValueError(f"Domain {label} total requires at least one species.")
    if mode == "explicit":
        if "value" not in config:
            raise ValueError(f"Explicit {label} total requires a value.")
        minimum = parse_rational(config["value"], label=f"Explicit {label} total")
    else:
        if "value" in config:
            raise ValueError("ideal_gas_minimum does not accept an explicit value.")
        pressure, temperature = parameters[symbols.pressure], parameters[symbols.temperature]
        if pressure.lower <= 0:
            raise ValueError("Ideal-gas minimum requires positive pressure.")
        minimum = pressure.lower / (GAS_CONSTANT * temperature.upper)
    return TotalConstraint(label, tuple(symbols.concentration(item) for item in selected), minimum)


def _domain(value, species, symbols, parameters):
    config = _mapping(value, "Case 'domain'")
    _keys(config, ("concentration_model", "upper", "excursion_lower", "totals"), "domain")
    try:
        model = ConcentrationModel(config.get("concentration_model"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Domain concentration_model must be 'independent' or 'chamfered'."
        ) from error
    ids = tuple(item.id for item in species)
    upper, lower = (
        _bounds(config.get("upper"), ids, "upper"),
        _bounds(config.get("excursion_lower"), ids, "excursion_lower"),
    )
    totals = _mapping(config.get("totals", {}), "Domain 'totals'")
    _keys(totals, ("gas", "solid"), "domain totals")
    constraints = tuple(
        item
        for phase in Phase
        if (item := _total(totals.get(phase.value), phase, species, symbols, parameters))
        is not None
    )
    return DomainSpec(
        symbols,
        model,
        {symbols.concentration(item): upper[item] for item in ids},
        {symbols.concentration(item): lower[item] for item in ids},
        parameters,
        constraints,
    )


def _checks(value):
    config = {} if value is None else _mapping(value, "Case 'checks'")
    _keys(config, ("profile", "include", "exclude", "fail_fast"), "checks")
    profile = config.get("profile", "physical")
    include, exclude = _strings(config, "include", True), _strings(config, "exclude", True)
    fail_fast = config.get("fail_fast", "stage")
    if fail_fast not in {"stage", "none"}:
        raise ValueError("Check fail_fast must be 'stage' or 'none'.")
    from .checks import plan_checks

    plan_checks(profile=profile, include=include, exclude=exclude)
    return {"profile": profile, "include": include, "exclude": exclude, "fail_fast": fail_fast}


def _report(value):
    config = {} if value is None else _mapping(value, "Case 'report'")
    _keys(config, ("verbosity", "format"), "report")
    verbosity, output = config.get("verbosity", "failures"), config.get("format", "text")
    if verbosity not in {"summary", "failures", "full"}:
        raise ValueError("Report verbosity must be 'summary', 'failures', or 'full'.")
    if output not in {"text", "json"}:
        raise ValueError("Report format must be 'text' or 'json'.")
    return {"verbosity": verbosity, "format": output}


def _selector(value):
    parts = value.split(".")
    if len(parts) not in (1, 2) or any(not part.isidentifier() for part in parts):
        raise ValueError(
            f"Invalid reaction selector '{value}'; expected 'family' or 'family.reaction'."
        )
    return parts[0], parts[1] if len(parts) == 2 else None


def _local_module(path, family_id):
    name = f"_rxn_checker_case_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]}_{family_id}"
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local reaction family '{family_id}'.")
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _family_module(directory, family_id):
    local = directory / "reactions" / f"{family_id}.py"
    if local.is_file():
        return _local_module(local, family_id)
    try:
        return import_module(BUILTIN_FAMILIES[family_id])
    except KeyError as error:
        raise ValueError(f"Unknown reaction family '{family_id}'.") from error


def _build_family(directory, family_id, symbols):
    builder = getattr(_family_module(directory, family_id), "build_family", None)
    if not callable(builder):
        raise TypeError(f"Reaction family '{family_id}' must define callable build_family().")
    built = builder(symbols)
    if not isinstance(built, Mapping) or not built:
        raise TypeError(f"Reaction family '{family_id}' must return a non-empty mapping.")
    for local_id, reaction in built.items():
        if not isinstance(local_id, str) or not local_id.isidentifier():
            raise ValueError(f"Reaction family '{family_id}' returned invalid id '{local_id}'.")
        if not isinstance(reaction, Reaction):
            raise TypeError(f"Reaction family '{family_id}' returned a non-Reaction value.")
        if reaction.id != f"{family_id}.{local_id}":
            raise ValueError(
                f"Reaction family '{family_id}' key '{local_id}' returned reaction id "
                f"'{reaction.id}', expected '{family_id}.{local_id}'."
            )
    return built


def _reactions(selectors, symbols, directory):
    parsed = tuple(map(_selector, selectors))
    families = {
        family_id: _build_family(directory, family_id, symbols)
        for family_id in dict.fromkeys(family for family, _ in parsed)
    }
    selected, selected_ids = [], set()
    for family_id, local_id in parsed:
        family = families[family_id]
        for name in family if local_id is None else (local_id,):
            if name not in family:
                raise ValueError(
                    f"Unknown reaction '{family_id}.{name}'. Available reactions: "
                    + ", ".join(family)
                    + "."
                )
            if family[name].id in selected_ids:
                raise ValueError(f"Reaction '{family[name].id}' was selected more than once.")
            selected.append(family[name])
            selected_ids.add(family[name].id)
    return tuple(selected)


def load_case(path: str | Path, *, property_registry: PropertyRegistry = PROPERTY_REGISTRY) -> Case:
    path = Path(path)
    path = path / "case.yaml" if path.is_dir() else path
    with path.open(encoding="utf-8") as stream:
        config = _mapping(yaml.safe_load(stream), "Case document")
    _keys(config, _TOP_KEYS, "top-level case")
    if config.get("schema") != 1 or isinstance(config.get("schema"), bool):
        raise ValueError("Case 'schema' must be integer 1.")
    ids, inerts, selectors = (
        _strings(config, "species"),
        _strings(config, "inerts", True),
        _strings(config, "reactions"),
    )
    missing = [item for item in ids if item not in property_registry.records]
    if missing:
        raise ValueError("Unknown case species: " + ", ".join(missing) + ".")
    species = tuple(property_registry.get_record(item) for item in ids)
    symbols = CaseSymbols.for_species(ids)
    parameters = _parameters(config.get("parameters"), symbols)
    return Case(
        path.parent.name,
        species,
        symbols,
        _reactions(selectors, symbols, path.parent),
        _domain(config.get("domain"), species, symbols, parameters),
        inerts,
        _checks(config.get("checks")),
        _report(config.get("report")),
    )
