# rxn-checker

Develop reaction rate expressions for numerical solvers. Checks use exact
stoichiometry, symbolic expressions, and domain bounds to identify invalid rates,
singularities, expensive evaluation, and difficult derivative behaviour.

```sh
uv sync
uv run rxn-checker example_case --profile all
uv run rxn-checker reforming_case --checks evaluation_profile,differential_solver_profile --format json
uv run rxn-checker --list-checks
```

Each case directory contains a `case.yaml`; see the two bundled cases for the
schema. Rates use SI units. Concentrations, temperature, and pressure have explicit
bounds. The physical domain has nonnegative concentrations; the augmented domain
also permits the configured negative excursions.

Profiles select checks and their prerequisites:

| Profile | Checks |
| --- | --- |
| `basic` | Atom and mass conservation |
| `physical` | Basic checks plus rate definedness, nonnegativity, depletion, Lipschitz bounds, and invariance |
| `robust` | Physical checks plus behaviour during negative excursions |
| `analysis` | Physical checks plus conserved quantities, structural faces, independent steady-state equations, and numerical profiles |
| `all` | Every check |

`--checks` selects specific checks with their prerequisites. `--skip` excludes
checks and their dependants. Reports go to stdout and `report.txt` or `report.json`
in the case directory. Exit codes are 0 for passing blocking checks, 1 for an
unproved or failed blocking check, and 2 for an error. `--debug` shows tracebacks.

To add kinetics, create `reactions/my_family.py` inside a case directory:

```python
from rxn_checker import Reaction


def build_family(symbols):
    a = symbols.concentration("Aye")
    return {"transfer": Reaction("my_family.transfer", {"Aye": 1}, {"Bee": 1}, (), 2 * a)}
```

Select `my_family` or `my_family.transfer` in the YAML `reactions` list. A local
family takes precedence over a built-in family with the same name. Only selected
families are imported; each is built once per load.

`UNKNOWN` means the symbolic analysis could not establish a result. Evaluation
counts describe expression work, not runtime. Differential profiles bound the
kinetic subsystem, not a full solver Jacobian or an allowed time step. These
profiles are nonblocking, and their JSON evidence retains partial results.

JSON schema 2 serializes numerical profile records directly. Evaluation evidence
has `declared` and `source_equivalent` views, each with `outputs`, `raw`, `cse`,
and `local_cse`. Differential evidence has `physical` and `augmented` records
with rates, matrix entries, bounds, and completeness flags. Symbolic values are
strings. This replaces the former duplicated report-specific layouts.

Python imports use `rxn_checker.checks` and `rxn_checker.species`. Compatibility
submodules and per-reaction forwarding functions have been removed; obtain rates
from a family's `build_family()` result. Custom `CheckSpec` objects use `role`
to indicate whether they block a run.

The source follows the workflow: `model.py`, `case.py`, and `domain.py` define the
inputs; `loading.py` reads cases; `checks/core.py` selects and runs checks;
`proof/` holds the mathematical analysis; `reporting.py` renders results.

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
