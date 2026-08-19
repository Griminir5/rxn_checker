# rxn-checker

`rxn-checker` validates symbolic reaction implementations. It currently checks
that selected reactions exist, use case-owned state symbols, reference declared
species, conserve atoms and mass, have non-negative rates in the physical
domain, and stop when a reactant or catalyst is depleted. It also certifies
rate regularity on physical and augmented domains when the required domain
guards can be proved.

## Running a case

A schema-1 case separates bounded parameters from its concentration domain:

```yaml
schema: 1

species:
  - Aye
  - Bee
  - Cee

reactions:
  - aye_to_bee.simple

parameters:
  temperature: [200.0, 1500.0]
  pressure: [10000.0, 10000000.0]

domain:
  concentration_model: independent
  upper:
    default: 1000.0
    overrides:
      Aye: 100.0
  excursion_lower:
    default: -0.1
```

A selector is either `family.reaction` for one implementation or `family` for
every implementation in that family. Overlapping selectors are rejected.

Run the checks against a case file or a directory containing `case.yaml`:

```shell
uv run rxn-checker example_case/case.yaml
uv run rxn-checker example_case --profile robust
uv run rxn-checker example_case --checks physical_lipschitz
uv run rxn-checker example_case --skip steady_state_equations
uv run rxn-checker example_case --format json --output report.json
uv run rxn-checker --list-checks
```

The command always prints its report. It writes a file only when `--output` or
`report.output` is configured. Exit codes are `0` when every selected blocking
check passes, `1` for a failed or inconclusive result, and `2` for invalid input
or an internal error.

Cases select the `physical` profile by default. Available profiles are `basic`,
`physical`, `robust`, `analysis`, and `all`. `checks.include` adds a check and
its transitive prerequisites, while `checks.exclude` removes work. Explicitly
including a check while excluding one of its prerequisites is an error.

## State and reactions

`load_case()` creates concentration coordinates separately from the bounded
temperature and pressure parameter symbols. Numeric configuration and
stoichiometry are converted to exact SymPy rationals from their decimal
spellings:

```python
from rxn_checker import AnalysisContext, load_case

case = load_case("example_case/case.yaml")
aye = case.symbols.concentration("Aye")
temperature = case.symbols.temperature
pressure = case.symbols.pressure

assert aye in case.symbols.concentration_symbols
assert temperature in case.symbols.parameter_symbols
assert case.domain_spec.parameter_intervals[temperature].lower == 200
assert case.domain_spec.upper[aye] == 100

context = AnalysisContext(case)
assert context.physical_domain.interval(aye).lower == 0
assert context.augmented_domain.interval(aye).lower == -0.1
```

`concentration_model` is either `independent` or `chamfered`. A chamfered domain
may configure one gas total and one solid total. Gas totals support `none`,
`explicit`, and `ideal_gas_minimum`; the last is the robust lower envelope
`pressure_min / (R * temperature_max)`, not an ideal-gas equality constraint.
Solid totals support `none` and `explicit`. Inerts remain ordinary non-negative
concentrations and may occur in a rate through dilution, but cannot participate
as reactants, products, or catalysts.

Reaction families are imported only when selected. Each module exposes one
family-level builder so shared expressions are constructed once:

```python
from sympy import Rational

from rxn_checker import CaseSymbols, Reaction


def build_family(symbols: CaseSymbols):
    aye = symbols.concentration("Aye")
    return {
        "simple": Reaction(
            id="aye_to_bee.simple",
            reactants={"Aye": Rational(1)},
            products={"Bee": Rational(1)},
            catalysts=(),
            rate=2 * aye,
        )
    }
```

Built-in families live in `rxn_checker.reactions`. A case may instead provide
`reactions/<family>.py` beside `case.yaml`; local reaction modules are trusted
Python code and execute when that family is selected. Builders must use the
supplied symbols. Reactant and product mappings remain separate, while exact
`net_stoichiometry` is computed once. Model each reaction direction as its own
`Reaction`; no reversible pairing is inferred.

## Checks and reports

Atom conservation reports per-element totals and product-minus-reactant
imbalances. Mass conservation reports reactant mass, product mass, and their
signed difference in kg/mol. Atom totals are exact rationals and use no
tolerance. Mass comparison retains configurable `rel_tol` and `abs_tol`
arguments because declared molar masses may be rounded. A missing molecular
weight fails the chemistry gate.

One context-owned expression analyzer supplies exact affine bounds, composed
interval bounds, sign proofs, and recursive definedness guards. It supports
ordinary arithmetic together with `Abs`, `Min`, `Max`, exponential, logarithmic,
trigonometric, and hyperbolic functions. Bounds are uniform over concentration,
temperature, and pressure intervals. Loose but sound bounds return `UNKNOWN`;
they are never treated as failures. A `FAIL` requires an exact feasible witness,
which is retained as structured report evidence.

Rate non-negativity uses this analyzer on the physical domain. Repeated
subexpressions and repeated requests for the same expression/domain pair reuse
memoized results. Expensive fallback work is bounded, with only a small
`factor_terms()` rewrite attempted for unresolved expressions.

Definedness is checked independently and reports its first failed or unresolved
guard, such as a denominator that can equal zero or a logarithm argument that
is not strictly positive. Unsupported discontinuous functions return `UNKNOWN`
instead of triggering unrestricted symbolic search.

The zero-at-depletion check independently sets every reactant and catalyst
case-owned concentration coordinate to zero and requires the resulting rate to
be exactly zero. Product-only species are not depletion boundaries. Undefined
forms such as `0/0` and disproven identities are `FAIL`; an identity the bounded
symbolic rules cannot decide is `UNKNOWN`.

Lipschitz certification uses the concentration-space infinity norm, treating
temperature and pressure as uniformly bounded parameters rather than state
coordinates. It assembles exact constants from elementary sum, product,
reciprocal, power, and supported-function inequalities. Every denominator or
noninteger power that needs separation from a singular boundary records its
strict guard margin. A bounded partial-derivative fallback handles small smooth
expressions outside the explicit rules.

When every rate passes, the checker derives component and source-vector bounds
directly from the stoichiometric matrix: it does not differentiate the expanded
network source. Text reports show the exact constant when compact and a numeric
approximation otherwise, while JSON retains exact and approximate per-rate
constants, active concentration variables, uniform parameters, and guard
margins. Negative-side non-repulsion remains a placeholder until its sparse
Phase 7 theorem is implemented.

Physical boundary inwardness is derived without expanding or reanalysing the
network source. On each zero-concentration face, consuming contributions vanish
by the depletion results and every remaining stoichiometric contribution is
non-negative by the rate results. Combining that certificate with the
source-vector Lipschitz certificate proves forward invariance of the
nonnegative orthant throughout the declared physical domain. This is not a
claim of persistence.

The runner produces one renderer-independent `RunResult`. Each `CheckResult`
contains `Finding` objects with one of these verdicts:

```text
PASS  FAIL  UNKNOWN  SKIPPED  ERROR
```

`UNKNOWN` means mathematical analysis was inconclusive. `SKIPPED` records a
failed prerequisite or absent configuration. Unexpected implementation
exceptions are `ERROR`, never `UNKNOWN`. Only blocking checks determine the
overall mathematical result; internal errors always produce an overall
`ERROR`.

Checks form a validated static DAG. Dependencies are expanded in deterministic
topological order and execute once even when selected through multiple paths.
Reaction-scoped checks reuse prerequisite findings per reaction, so one
undefined rate does not suppress conclusions for unrelated reactions.
The default `fail_fast: stage` completes the current stage, then skips later
stages after a blocking failure. In particular, atom and mass failures complete
the chemistry gate before symbolic work stops.

To add a check, define a runner and its metadata in a module:

```python
from rxn_checker.checks import CheckScope, CheckSpec, Stage
from rxn_checker.results import Finding, Role, Verdict


def run(context, dependencies):
    passed = prove_property(context.physical_domain)
    return Finding(
        context.case.name,
        Verdict.PASS if passed else Verdict.UNKNOWN,
        "Property proved." if passed else "Proof was inconclusive.",
    )


CHECK = CheckSpec(
    id="example_property",
    name="Example property",
    stage=Stage.PHYSICAL,
    scope=CheckScope.CASE,
    requires=("physical_rate_definedness",),
    blocking=False,
    role=Role.ADVISORY,
    profiles=frozenset(("all",)),
    run=run,
)
```

Add the explicit specification to `checks/registry.py`. The registry is checked
for duplicate ids, missing dependencies, later-stage dependencies, and cycles
at import time. Text and JSON renderers consume the same structured findings.

## Development

```shell
uv sync
uv run pytest -q
```
