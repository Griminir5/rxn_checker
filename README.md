# rxn-checker

`rxn-checker` validates symbolic reaction implementations. It currently checks
that selected reactions exist, use case-owned state symbols, reference declared
species, conserve atoms and mass, have non-negative rates in the physical
domain, and stop when a reactant or catalyst is depleted.

## Running a case

A case lists species and reaction selectors:

```yaml
species:
  - Aye
  - Bee
reactions:
  - aye_to_bee.simple
```

A selector is either `family.reaction` for one implementation or `family` for
every implementation in that family. Overlapping selectors are rejected.

Run the checks against a case file or a directory containing `case.yaml`:

```shell
uv run rxn-checker example_case/case.yaml
uv run rxn-checker example_case
```

The command prints a report and writes `rxn-checker-report.txt` beside the case
file. Exit codes are `0` for a successful report, `1` for a failed or
inconclusive report, and `2` when the case cannot be loaded or the report cannot
be written.

## State and reactions

`load_case()` creates the case's concentration, temperature, and pressure
symbols:

```python
from rxn_checker import load_case

case = load_case("example_case/case.yaml")
aye = case.states.concentration("Aye")
temperature = case.states.temperature
pressure = case.states.pressure
```

Concentrations are real but not assumed non-negative, allowing later checks to
inspect expressions just outside the physical domain. Loading rejects unknown
species or reactions, duplicate selections, missing reaction species, and rate
symbols not owned by the case.

Each module in `rxn_checker.reactions` is an automatically discovered reaction
family. It exposes a `REACTIONS` mapping from local names to builder functions:

```python
from rxn_checker import Reaction


def build_simple(states):
    aye = states.concentration("Aye")
    return Reaction(
        name="simple",
        family="aye_to_bee",
        reactants={"Aye": 1},
        products={"Bee": 1},
        rate=2.0 * aye,
    )


REACTIONS = {"simple": build_simple}
```

Builders receive the case's state object, so reaction modules should not create
their own SymPy symbols. A reaction's qualified id is derived as `family.name`.
Its reactant and product mappings remain separate, while `net_stoichiometry` is
derived. Catalysts are non-consumed species and therefore do not participate in
atom or mass balances. Model each direction of a reversible reaction as a
separate `Reaction`.

## Checks and reports

Atom conservation reports per-element totals and product-minus-reactant
imbalances. Mass conservation reports reactant mass, product mass, and their
signed difference in kg/mol. Both checks accept `rel_tol` and `abs_tol`
overrides. Missing molecular weights make the registered mass check
`UNAVAILABLE` rather than failed.

Rate non-negativity asks SymPy to prove that an expression is non-negative when
all of its state variables are non-negative. A symbolic proof is a `PASS`, and
a rate proven negative in the positive interior is a `FAIL`. Other expressions
remain `INDETERMINATE`; a later interval branch-and-bound check will use ranges
specified by the case to resolve them.

The zero-at-depletion check independently sets every reactant and catalyst
concentration to zero and requires the resulting symbolic rate to be exactly
zero. Product-only species are not depletion boundaries. A disproven identity
is a `FAIL`, while an identity SymPy cannot decide is `INDETERMINATE`.

Checks return `CheckOutcome` objects with an optional qualitative status,
details, and numerical values. Supported statuses, from least to most
consequential, are:

```text
PASS < SAMPLED_PASS < UNAVAILABLE < INDETERMINATE < FAIL
```

Numerical-only outcomes need no status. An unexpected exception or invalid
return is isolated as `INDETERMINATE`, and later checks still run.

To add a check, define a runner and its metadata in a module:

```python
from .models import (
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)


def run(case, context):
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

Add the module's `CHECK` to `checks/registry.py`. Registry order controls report
order; the runner and renderer do not need to change. A reaction-wide runner can
return one outcome per reaction with `subject` set to the reaction id.

## Development

```shell
uv sync
PYTHONPATH=src .venv/bin/python -m unittest discover -v
```
