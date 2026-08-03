# rxn-checker

`rxn-checker` checks symbolic reaction implementations for physical and
numerical robustness.

The current foundation has three small pieces:

- `rxn_checker.species` stores species properties.
- `rxn_checker` loads a case and owns all symbolic state variables.
- `rxn_checker.reactions` defines one-way reactions using those state variables.

## Case-owned state variables

A case lists its species and reaction selectors:

```yaml
species:
  - Aye
  - Bee
reactions:
  - aye_to_bee.simple
```

A selector has one of two forms:

- `family.reaction` selects one reaction from a family;
- `family` selects every reaction in that family.

For example, replacing `aye_to_bee.simple` above with `aye_to_bee` selects
both `aye_to_bee.simple` and `aye_to_bee.autocatalytic`.

`load_case()` creates one `StateVariables` object containing the concentration
symbol for every listed species, plus temperature and pressure:

```python
from rxn_checker import load_case

case = load_case("example_case/case.yaml")

aye = case.states.concentration("Aye")
bee = case.states.concentration("Bee")
temperature = case.states.temperature
pressure = case.states.pressure
```

Concentration symbols are real but are not assumed non-negative, because later
checks need to inspect expressions just outside the physical domain.

## Reaction family modules

The family part of a selector maps directly to a same-named packaged module.
For example, both `aye_to_bee` and `aye_to_bee.simple` load
`rxn_checker/reactions/aye_to_bee.py`. Each family module exposes a `REACTIONS`
mapping from its local reaction names to builder functions:

```python
from types import MappingProxyType

from rxn_checker import Reaction

RATE_CONSTANT = 2.0


def build_simple(states):
    aye = states.concentration("Aye")

    return Reaction(
        name="simple",
        family="aye_to_bee",
        reactants={"Aye": 1},
        products={"Bee": 1},
        rate=RATE_CONSTANT * aye,
    )


REACTIONS = MappingProxyType({"simple": build_simple})
```

The package automatically discovers these family modules and combines their
local mappings into a unified builder registry. There is no central list to
maintain: adding a family file is enough to register it. Discovery imports the
family modules and records their builder functions, but it does not build any
`Reaction` objects. A qualified selector calls one builder, while a bare family
selector calls every builder in that family's mapping order.

The case passes its state object into each selected builder, so reaction code
does not create SymPy symbols. Constants and correlations stay as ordinary
numbers and Python code inside the reaction module. Consequently, every free
SymPy symbol in a rate represents a state variable.

After building a reaction, the loader verifies its fully qualified id and
family. `Case` then verifies that its species exist and every free rate symbol
belongs to the case. Overlapping selectors such as `aye_to_bee` together with
`aye_to_bee.simple` are rejected rather than loading a reaction twice.

## Reaction definition

A `Reaction` represents one forward direction and contains:

- a local reaction name;
- a family id for implementations of the same net reaction;
- separate reactant and product mappings;
- optional non-consumed catalysts;
- the resulting SymPy rate expression.

Its globally unique `id` is derived as `family.name`, so a family implementation
does not repeat the qualified selector manually.

Net stoichiometry is derived. The authoritative sided mappings preserve the
difference between these mechanisms:

```text
Aye + Bee -> 2 Bee
Aye       -> Bee
```

Both have net `Aye: -1, Bee: +1`, but their rate expressions and boundary
behaviour are different. A reversible expression is not represented by one
reaction; each direction must be a separate `Reaction`.

Species existence, atom balance, family comparison, and physical/numerical
properties are later checks. They are intentionally not responsibilities of
the species registry or reaction-definition object.

## Conservation checks

Unknown requested species and reactions are rejected while loading a case,
and `Case` rejects reactions whose reactants, products, or catalysts are not
present. Atom and mass conservation are explicit checks so that an invalid
reaction can be loaded and reported rather than preventing the whole case from
being inspected:

```python
from rxn_checker import check_atom_conservation, check_mass_conservation

for reaction in case.reactions:
    atom_result = check_atom_conservation(reaction)
    mass_result = check_mass_conservation(reaction)
    print(reaction.id, atom_result.passed, mass_result.passed)
```

Atom results contain the per-element totals on both sides and `imbalances`,
defined as products minus reactants. Mass results contain `reactant_mass`,
`product_mass`, and the signed `imbalance`, all using the property registry's
kg/mol units. The mass check uses a small relative tolerance because tabulated
molecular weights are rounded; both checks accept `rel_tol` and `abs_tol`
overrides. A missing molecular weight raises `ValueError`, distinguishing an
unavailable check from a failed conservation check. Catalysts are excluded
from both balances because they are non-consumed by definition. The registered
mass check translates a missing molecular weight into an `UNAVAILABLE`
outcome.

## Command-line report

Run every registered check by pointing `rxn-checker` at either a case YAML
file or its directory:

```shell
rxn-checker example_case/case.yaml
# or
rxn-checker example_case
```

The command prints a plain-text report and writes the same report to
`rxn-checker-report.txt` in the case directory. Its exit code is `0` when the
overall status is `PASS` or `SAMPLED_PASS` (or the report contains only
numerical results), `1` for `FAIL`, `INDETERMINATE`, or `UNAVAILABLE`, and `2`
when the case cannot be loaded or the report cannot be written.

## Adding checks

Each check lives in its own module, accepts the complete `Case` plus a shared
`CheckContext`, and returns either one `CheckOutcome` or an iterable of them.
A case-wide check normally returns one outcome. A reaction-wide check can loop
over `case.reactions` and return one outcome per reaction, setting `subject` to
the reaction id.

For example, a sampled case-wide check with numerical diagnostics has this
shape:

```python
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)


def run(case, context: CheckContext):
    minimum_rate, sample_count = sample_rates(case)
    return CheckOutcome(
        status=CheckStatus.SAMPLED_PASS,
        details=("No negative rates were sampled.",),
        values=(
            CheckValue("Minimum rate", minimum_rate, "mol/m^3/s"),
            CheckValue("Samples", sample_count),
        ),
    )


CHECK = CheckDefinition(
    id="nonnegative_rates",
    name="Non-negative rates",
    group="Physical checks",
    scope=CheckScope.CASE,
    run=run,
)
```

Import that module's `CHECK` in `checks/registry.py` and append it to
`CHECK_REGISTRY`. Registry order controls report order; neither the CLI nor the
report renderer needs to change.

The supported qualitative statuses are `PASS`, `SAMPLED_PASS`, `FAIL`,
`INDETERMINATE`, and `UNAVAILABLE`. Omit `status` for a purely numerical
outcome. Values can also accompany a status. Overall status uses the most
consequential outcome in this order:

```text
FAIL > INDETERMINATE > UNAVAILABLE > SAMPLED_PASS > PASS
```

An unexpected exception or invalid return from one runner is isolated as an
`INDETERMINATE` outcome, and the remaining checks still run. Expected missing
prerequisites should be returned explicitly as `UNAVAILABLE`.

## Development

```shell
PYTHONPATH=src .venv/bin/python -m unittest discover -v
```
