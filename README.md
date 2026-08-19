# rxn-checker

`rxn-checker` validates symbolic reaction implementations. It currently checks
that selected reactions exist, use case-owned state symbols, reference declared
species, conserve atoms and mass, have non-negative rates in the physical
domain, stop when a reactant or catalyst is depleted, and form a positive
reaction network. It also applies two independent symbolic checks to declared
small negative concentration excursions and reports the conserved
stoichiometric quantities, symbolic equilibrium families, and terminal or
invariant concentration faces.

## Running a case

A case lists species, reaction selectors, and state bounds:

```yaml
species:
  - Aye
  - Bee
  - Cee
inerts:
  - Cee
reactions:
  - aye_to_bee.simple
bounds:
  temperature: [200.0, 1500.0]
  pressure: [10000.0, 10000000.0]
  concentrations:
    default:
      upper: 1000.0
      excursion_lower: -0.1
    overrides:
      Aye:
        upper: 100.0
gas_closure:
  minimum_total: ideal_gas
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

temperature_bounds = case.state_bounds[temperature]
aye_bounds = case.state_bounds[aye]
assert temperature_bounds.interval() == (200.0, 1500.0)
assert aye_bounds.interval() == (0.0, 100.0)
assert aye_bounds.interval(include_excursion=True) == (-0.1, 100.0)
```

Gas species also receive the ideal-gas closure
`pressure = total_gas_concentration * R * temperature`. The Lipschitz check's
augmented domain uses the selected lower-total policy. `minimum_total:
positive` assumes only that the gas total is strictly positive, leaving zero as
a limiting boundary. `minimum_total: ideal_gas` derives the uniform lower bound
`pressure_min / (R * temperature_max)`. Individual gas concentrations retain
their independent bounds and negative excursions. Solid concentrations receive
no corresponding total constraint. Omitting `gas_closure` defaults to
`minimum_total: positive`.

Concentrations are real but not assumed non-negative, allowing later checks to
inspect expressions just outside the physical domain. Their physical lower
bound is zero; species listed under `inerts` use the strict physical condition
`concentration > 0`. An inert must be a case species and cannot be a reactant,
product, or catalyst in any selected reaction, though it may still affect a
rate through dilution. `excursion_lower` defines how far the recovery check may
inspect the unphysical region. Concentration defaults apply
to every species and entries under `overrides` replace either value for one
species. Loading rejects invalid bounds, unknown species or reactions, duplicate
selections, participating inerts, missing reaction species, and rate symbols
not owned by the case.

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

Rate non-negativity asks SymPy to prove that an expression is non-negative using
sign assumptions implied by each state's physical lower bound. A zero lower
bound makes a symbol non-negative, while a positive lower bound makes it
strictly positive. A symbolic proof is a `PASS`, and a rate proven negative in
the physical interior is a `FAIL`. Upper bounds are reserved for a later
interval branch-and-bound check that can resolve otherwise indeterminate rates.

The Lipschitz-continuity check examines each reaction rate separately on the
domain formed by the declared upper bounds and excursion lower bounds,
intersected with strictly positive total gas concentration. It tests strict
domain conditions on the closure of that chamfer, including the zero-total
boundary, so a pass certifies a uniform Lipschitz margin on an open
neighbourhood. Expressions such as the inverse or square root of total gas
concentration therefore fail when their behavior becomes unbounded at the
excluded boundary. Checking rates rather than `S r` prevents singularities in
different reactions from being hidden by source-term cancellation. Polynomial,
rational, exponential, logarithmic, absolute-value, minimum, and maximum
expressions are supported; other functions are reported as `INDETERMINATE`.

The zero-at-depletion check independently sets every reactant and catalyst
concentration to zero and requires the resulting symbolic rate to be exactly
zero. Product-only species are not depletion boundaries. A disproven identity
is a `FAIL`, while an identity SymPy cannot decide is `INDETERMINATE`.

The equilibrium check constructs the complete physical steady-state
relationship instead of asking a general solver for explicit roots. It selects
a low-complexity basis of the stoichiometric source rows, applies identities
proved by the configured physical bounds, retains denominator conditions,
names repeated algebraic values such as roots, and splits shared zero factors
into exact alternatives. The result is read branch by branch: helper
definitions first, balance equations second, then nonzero and sign conditions.

The check is part of the default report. Its structured result is also
available directly. The text report prints the complete branch equations in a
width-limited form: repeated expressions receive short names, long sums place
one term on each line, and helper definitions appear before the balances that
use them.

```python
from rxn_checker import check_equilibria, load_case


case = load_case("xu_froment_case/case.yaml")
relation = check_equilibria(case)

for branch in relation.branches:
    print(branch.label)
    for helper in branch.helpers:
        print(f"  define {helper.symbol}: 0 = {helper.equation}")
    for balance in branch.balances:
        print(f"  0 = {balance}")
    for expression in branch.nonzero:
        print(f"  require {expression} != 0")
    for condition in branch.conditions:
        print(f"  require {condition}")
```

For algebraic rate laws, every branch is an exact constrained polynomial
system. Concentration-dependent transcendental functions are retained as exact
helper definitions and the result is marked mixed rather than pretending that
polynomial elimination applies. “Equilibrium” here means a kinetic steady
state, `S r = 0`; it does not require every reversible pair to be individually
balanced.

The repository also includes a standalone Xu–Froment equilibrium-surface toy
at `plots/xu_froment_equilibrium.html`. It fixes a stoichiometric
compatibility class from an editable reference mixture, numerically continues
the Ni-positive steady state over a temperature/pressure grid, and renders six
linked, dependency-free wireframe plots. Open the file directly in a browser;
no local server or JavaScript packages are required.

The terminal-face check does not globally simplify expressions. It incrementally
restricts shared rate expressions, reuses symbolic sign facts and exact interior
witness points, and branches only on concentration dependencies that can make a
disproved source identity vanish. A terminal face makes every component of `F`
vanish, while an invariant face only makes the depleted species' components
vanish. Only maximal faces are reported. Dense adversarial expressions can
still require combinatorial search, which stops as `INDETERMINATE` after 4096
distinct face tests.

The conserved-quantity analysis builds the case stoichiometric matrix exactly
and finds its left nullspace. It reports individually unchanged species,
analyzes disconnected reaction-network components separately, and presents the
non-negative extreme rays of each component's conservation cone. Signed basis
relations are included when those rays do not span the complete nullspace. The
extreme rays are constructed by an exact incremental double-description
algorithm rather than by enumerating candidate species supports.
These expressions are conserved by the selected reaction source terms; external
flows, dilution, and changing volume are outside the analysis. Cases do not yet
define initial concentrations, so the report gives expressions rather than
their constant numerical values.

The nonphysical-recovery check constructs the complete network source
`F = S r` and examines every nonempty negative-species set within each
stoichiometric component. Non-negative conservation rays exclude sign-regions
whose compatibility classes cannot reach the physical orthant. On every
remaining region it checks that rates and source terms are real and finite,
classifies the normalized restoration score, reports the stronger
componentwise result, and verifies every lower excursion face. Results use the
verdicts `STRONGLY_RESTORING`, `NET_RESTORING`, `NON_WORSENING`, `STUCK`,
`WORSENING`, `UNDEFINED_IN_EXTENSION`, `STOICHIOMETRICALLY_UNREPAIRABLE`, and
`INDETERMINATE`. All proofs and counterexamples are exact; there is no numerical
sampling. A bounded region count and symbolic-operation limit make large
expressions terminate conservatively as `INDETERMINATE`. The public
`check_nonphysical_recovery()` function performs the full bounded enumeration
by default. The registered report check may stop earlier after an exact failure
certificate, because later regions cannot change the case-level `FAIL` result.
A restoring verdict does not claim finite-time re-entry into the physical
domain.

The separate negative-side-recovery check is componentwise and does not use
conservation rays or restrict the state to a stoichiometric compatibility
class. For every concentration with a declared negative excursion, it asks on
the complete augmented state domain whether
`x_i <= 0` implies `f_i(x) >= 0`, where `F = S r`. Every other concentration
retains its own possible negative excursion, so simultaneous solver errors are
included. A `PASS` certifies this non-repulsion implication for every checked
species. The check also attempts the stronger implication `x_i < 0` implies
`f_i(x) > 0`; strict attraction is reported per species but is not required for
a non-worsening pass. Neither conclusion claims finite-time return to zero.

This check deliberately leaves rate definedness and physical-boundary
invariance to the independent checks. In conjunction with Lipschitz continuity
on the augmented domain, zero rate at every reactant or catalyst depletion
boundary, and non-negative rates on the physical domain, its `PASS` certifies
that the reaction source cannot drive a small negative concentration farther
from the physical domain. Where strict attraction is also proved, that
component improves while it remains negative.

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
