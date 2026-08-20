# Verdict

Your impression is correct. The refactor improved the architecture and replaced several bad algorithms, but it did **not** reduce the production source. Relative to the last pre-refactor commit, the current `src/rxn_checker` is about **437 lines larger**, while the refactor also introduced roughly **1,600 lines of tests** and a **2,200-line `refactor.md`**. The current master snapshot is commit `6d98291`.

The main reason is not the mathematics. The old large symbolic modules were removed, but their place was taken by:

* an over-general check execution framework;
* several parallel proof-result types;
* repeated validation at every layer;
* manually assembled evidence payloads;
* separate wrappers for every small check;
* three different implementations of witness/counterexample search;
* optional analyses that still carry a fair amount of machinery.

I think a **30–35% reduction in production code is realistic without losing meaningful functionality**. A **40–45% reduction** is plausible if you simplify the public result/evidence API. A true **50% reduction** probably requires removing or postponing optional analyses, the derivative fallback, or some of the detailed JSON certificates.

Nested functions are not the main offender. There are a handful that could be flattened, but they account for little of the size. The real issue is duplicated responsibility.

---

# What the current size is buying you

The largest areas are approximately:

| Area                                      | Current size | Sensible target | Likely saving |
| ----------------------------------------- | -----------: | --------------: | ------------: |
| Expression and Lipschitz proof engine     | ~1,165 lines |         650–750 |       400–500 |
| Domain and YAML loading                   |     ~850–900 |         500–560 |       300–380 |
| Registry, planning, runner, prerequisites |     ~550–600 |         220–280 |       300–350 |
| Negative-side recovery                    |          483 |         180–230 |       250–300 |
| Core check wrappers                       |     ~850–950 |         450–550 |       350–450 |
| Model, case, species registry             |     ~500–550 |         300–350 |       180–240 |
| Results, reporting, CLI                   |     ~420–450 |         250–300 |       130–200 |
| Optional analyses                         |          379 |         250–300 |        80–130 |
| Built-in reaction families                |     ~650–700 |         480–550 |       120–180 |

Those estimates suggest around **2,000–2,500 production lines can be removed**, depending on how aggressively you simplify the result and analysis layers.

The proof engine itself is necessarily substantial, but there is a lot of mechanical duplication inside it. `analysis.py` contains separate result structures for bounds, signs, sign proofs and definedness, threads `active_variables` through operations that do not use it, and maintains several closely related caches.

---

# 1. Collapse the proof result model

This is the highest-leverage change.

At present, the proof layer has:

```python
BoundResult
SignResult
SignProof
DefinednessResult
LipschitzResult
LipschitzCertificate
GuardMargin
NetworkLipschitzCertificate
```

Many fields repeat:

```text
verdict
decisive_subexpression
witness
reason
bounds
witness_value
```

A smaller core could be:

```python
@dataclass(frozen=True)
class Bounds:
    lower: Expr | None
    upper: Expr | None
    exact: bool = False
    witness_lower: Point | None = None
    witness_upper: Point | None = None


@dataclass(frozen=True)
class Proof:
    verdict: Verdict
    bounds: Bounds | None = None
    value: object | None = None
    culprit: Expr | None = None
    witness: Point | None = None
    reason: str | None = None
```

Then:

```python
analyzer.bounds(expr, domain) -> Bounds
analyzer.sign(expr, domain) -> Sign
analyzer.prove(expr, domain, NONNEGATIVE) -> Proof
analyzer.defined(expr, domain) -> Proof
analyzer.zero(expr, domain) -> Proof
analyzer.lipschitz(expr, domain) -> Proof
```

The proof object’s `value` can hold:

* a `Sign`;
* a Lipschitz estimate;
* a zero conclusion;
* a guard margin.

You do not need a public dataclass for every variant.

## Remove `active_variables` from most APIs

`active_variables` is included in the cache key and recursively passed through `bounds`, `sign`, `prove_sign`, and `defined`, but these calculations do not conceptually depend on which variables are regarded as Lipschitz coordinates. Only the Lipschitz calculation needs that distinction.

Use:

```python
analyzer.bounds(expr, domain)
analyzer.sign(expr, domain)
analyzer.defined(expr, domain)
analyzer.lipschitz(expr, domain, variables)
```

In practice, all current rate checks use every concentration coordinate anyway. The context could own that tuple once:

```python
context.concentrations
```

and `lipschitz()` would not need a caller-supplied variable list at all.

## Use the domain object as the cache key

The analyzer currently keys by `id(domain)` and retains a separate `_domains` dictionary to prevent identity reuse. That is clever but unnecessarily indirect.

Make the domain identity-hashable:

```python
@dataclass(frozen=True, eq=False)
class Domain:
    ...
```

Then cache on:

```python
(expression, domain)
```

No `_domains` dictionary is needed.

## Add two generic methods

Two missing general methods are responsible for substantial duplication elsewhere:

```python
analyzer.prove_zero(expression, domain)
analyzer.prove_sum(terms, domain, requirement)
```

`prove_zero()` would replace most of `zero_at_depletion.py`.

`prove_sum()` would accept sparse terms such as:

```python
[(coefficient, rate_expression), ...]
```

and perform:

1. interval bounds for each term;
2. summation of known lower and upper bounds;
3. assembly of only the unresolved residue;
4. bounded exact witness search.

That would replace most of `negative_side.py`.

---

# 2. Make the Lipschitz recursion return a tiny internal object

The public Lipschitz certificate is fine. The problem is that every recursive subexpression currently constructs a complete certificate containing:

* domain kind;
* norm name;
* active variables;
* uniform parameters;
* guard margins;
* constant.

That metadata is largely identical throughout the expression tree.

Use a small internal result:

```python
@dataclass(frozen=True)
class LipBound:
    constant: Expr
    guards: tuple[Guard, ...] = ()
```

or:

```python
LipBound | Failure
```

Only the top-level `lipschitz()` method should wrap it in:

```python
LipschitzCertificate(
    domain=...,
    norm="L_inf",
    variables=...,
    parameters=...,
    constant=...,
    guards=...,
)
```

## Collapse unary-function handling

The current file has individual branches for:

* `sin`;
* `cos`;
* `tanh`;
* `atan`;
* `exp`;
* `log`;
* `sinh`;
* `cosh`.

Most are the same chain rule:

[
L(g\circ f)\le \sup_D|g'(f)|L(f).
]

Use one helper:

```python
def unary_lipschitz(expr, child, derivative_bound, guard=None):
    ...
```

With a small map or `match`:

```python
match expr.func:
    case sp.sin | sp.cos | sp.tanh | sp.atan:
        derivative_bound = 1
    case sp.exp:
        derivative_bound = exp(bounds(child).upper)
    case sp.log:
        guard = positive(child)
        derivative_bound = 1 / guard.margin
    ...
```

Keep special implementations only for:

* `Pow`;
* `Abs`;
* `Min`;
* `Max`;
* `Add`;
* `Mul`.

The current detailed implementation is mathematically reasonable, but its repeated construction and error handling make it almost twice as long as necessary.

## Do not over-optimize the product rule

The product implementation identifies “varying indices”, calculates exactly which factor bounds are required, and stores them in an indexed dictionary.

For the scale of this project, this optimization costs more cognitive complexity than it saves. Just bound all factors once:

```python
bounds = [absolute_bound(arg) for arg in args]
constant = sum(
    lip[i] * prod(bounds[:i] + bounds[i + 1 :])
    for i in range(len(args))
)
```

The analyzer cache means repeated bound requests are cheap.

I would retain the derivative fallback, but make it a clearly isolated final branch. Removing it would save additional code, but it is useful for keeping arbitrary SymPy functions possible.

---

# 3. Delete most of `negative_side.py`

This module is the most obvious case where check-specific code duplicates generic proof machinery.

It currently defines:

* `ContributionBound`;

* `SparseSourceResult`;

* custom term-bound propagation;

* custom exact candidate generation;

* custom sign-violation tests;

* custom residual construction;

* custom exact source evaluation;

* manual evidence conversion;

* manual strict-attraction labels;

* a special summary finding.

After adding `ExpressionAnalyzer.prove_sum()`, the check should be close to:

```python
def run(context, dependencies):
    findings = []

    for species, source_terms in context.network.sources.items():
        symbol = context.case.symbols.concentration(species)
        if context.augmented_domain.interval(symbol).lower >= 0:
            continue

        unavailable = nonregular_contributors(source_terms, dependencies)
        if unavailable:
            findings.append(skipped(species, unavailable))
            continue

        nonpositive = context.augmented_domain.restrict(symbol, upper=0)
        negative = context.augmented_domain.restrict(
            symbol, upper=0, strict_upper=True
        )

        inward = context.analyzer.prove_sum(
            source_terms,
            nonpositive,
            NONNEGATIVE,
        )
        attracting = context.analyzer.prove_sum(
            source_terms,
            negative,
            POSITIVE,
        )

        findings.append(recovery_finding(species, inward, attracting))

    return findings
```

The generic proof object can already contain:

* source bounds;
* unresolved expression;
* exact counterexample;
* diagnostic.

There is no need to copy every contribution into a separate nested evidence dictionary. The reaction network and proof already contain that information.

I would expect this module to fall from **483 lines to about 180–220**, while becoming easier to understand.

---

# 4. Replace `zero_at_depletion.py` with a thin loop

The current module is nearly 200 lines because it independently implements:

* exact-zero classification;
* numerator/denominator handling;
* factor fallback;
* witness generation;
* a specialized result dataclass;
* mapping proxies;
* result-to-finding conversion.

With `analyzer.prove_zero()`:

```python
def run(context, dependencies):
    findings = []

    for reaction in context.case.reactions:
        if skipped := reaction_skip(...):
            findings.append(skipped)
            continue

        proofs = {
            species: context.analyzer.prove_zero(
                reaction.rate.subs(
                    context.case.symbols.concentration(species), 0
                ),
                context.physical_domain.restrict(
                    context.case.symbols.concentration(species),
                    lower=0,
                    upper=0,
                ),
            )
            for species in required_species(reaction)
        }
        findings.append(depletion_finding(reaction.id, proofs))

    return findings
```

That should be **60–90 lines**, including user-facing findings.

---

# 5. Make the check DAG static rather than generic

The check registry and runner are designed as though this were a check-plugin framework for third parties. It is not.

The static registry currently stores profile membership on every check, then validates the registry, topologically sorts it, expands profile membership, computes dependency closure, detects exclusions, and cleans up invalid downstream selections.

## Define profiles by their terminal checks

Instead of putting profile sets on all fourteen checks:

```python
PROFILES = {
    "basic": (
        "atom_conservation",
        "mass_conservation",
    ),
    "physical": (
        "forward_invariance",
    ),
    "robust": (
        "forward_invariance",
        "negative_side_nonrepulsion",
    ),
    "analysis": (
        "forward_invariance",
        "conserved_quantities",
        "structural_faces",
        "steady_state_equations",
    ),
    "all": (
        "forward_invariance",
        "negative_side_nonrepulsion",
        "conserved_quantities",
        "structural_faces",
        "steady_state_equations",
    ),
}
```

Dependency closure automatically brings in all prerequisites.

This removes:

* `profiles` from `CheckSpec`;
* `_BASIC`, `_PHYSICAL`, `_ROBUST`, `_ANALYSIS`;
* profile validation per check;
* a lot of repeated registry data.

## Remove `Role` versus `blocking`

`CheckSpec` currently has both:

```python
blocking: bool
role: Role | None
```

and validates that they agree.

Keep one:

```python
blocking: bool
```

An analysis is simply nonblocking. `Role.ADVISORY` is not currently buying much.

## Stop validating the static registry repeatedly

The registry is currently validated:

* explicitly at import;
* during planning;
* again during execution.

The runner then also checks:

* duplicate IDs;
* missing dependencies;
* context ownership;
* selected-set consistency;
* prerequisite result shape.

Validate the registry once in a small test. `plan_checks()` can assume the built-in registry is correct. `execute_plan()` can assume it received a valid plan from `plan_checks()`.

This tool has one built-in registry maintained in the same repository. Runtime self-defence against the developer having constructed an invalid registry is not worthwhile.

## Remove custom-registry support

The following parameters make the framework much more general:

```python
checks: Iterable[CheckSpec] | None
registry: Iterable[CheckSpec]
context: AnalysisContext | None
```

Unless you actively use custom checks from external Python code, remove them.

A smaller interface is:

```python
run_checks(
    case,
    *,
    profile=None,
    only=(),
    skip=(),
    debug=False,
)
```

## Make stage failure fixed behaviour

You originally wanted atom or mass failure to terminate the analysis. I would encode that as behaviour, not a `fail_fast` configuration option.

```text
Chemistry fails → stop blocking physical/augmented stages.
```

Removing `fail_fast: stage|none` simplifies:

* YAML loading;
* case configuration;
* runner state;
* documentation;
* tests.

If you later need all checks regardless of chemistry, a `--force` developer option can be added, but I would not preserve generality pre-emptively.

A combined `checks/core.py` containing the specification, profiles, dependency closure and execution could be around **220–280 lines**, replacing the current `definitions.py`, `registry.py`, `runner.py`, and `prerequisites.py`, which together are substantially larger.

---

# 6. Flatten the result and evidence model

The current results layer wraps every data payload in:

```python
Evidence(kind, data)
```

then wraps that in:

```python
Finding(subject, verdict, summary, evidence)
```

and protects mappings through `MappingProxyType`.

Use:

```python
@dataclass(frozen=True)
class Finding:
    subject: str
    verdict: Verdict
    summary: str
    details: object | None = None
```

`details` can directly be:

* a `Proof`;
* a `LipschitzCertificate`;
* an imbalance dictionary;
* a conserved quantity;
* `None`.

A generic JSON serializer can handle:

```python
dataclasses.is_dataclass(value)
Enum
SymPy Basic
Mapping
Sequence
```

There is no need for every check to manually convert its internal results into nested dictionaries.

## Example

Instead of this style:

```python
Evidence(
    "negative_side_certificate",
    {
        "domain": "augmented",
        "coordinate": str(symbol),
        "non_repulsion": {
            "verdict": main_label,
            **_result_data(non_repulsion),
        },
        ...
    },
)
```

use:

```python
Finding(
    species_id,
    verdict,
    summary,
    RecoveryProof(non_repulsion, attraction),
)
```

Then the JSON renderer serializes `RecoveryProof`.

This removes both code and the possibility of the human summary and evidence dictionary disagreeing.

## Simplify text reporting

`render_text()` currently contains special behaviour for:

* reaction-scoped versus case-scoped findings;
* passing case details;
* aggregate reaction counts;
* failure filtering;
* analyses;
* a special `negative_side_summary` evidence kind.

A simpler rule is enough:

1. print one line per check;
2. reaction check with all passes: print `n/n`;
3. otherwise print every non-pass finding;
4. `full` verbosity additionally prints passing findings;
5. analyses print their findings.

The reporter should never know what a `negative_side_summary` is.

The CLI itself is not especially bad, but it can become smaller once checks and report configuration are typed and simpler.

---

# 7. Choose one validation boundary

Validation currently happens in too many places:

* YAML loader;
* `Species.__post_init__`;
* `CaseSymbols.__post_init__`;
* `Reaction.__post_init__`;
* `DomainSpec.__post_init__`;
* `Case.__post_init__`;
* check runner;
* check prerequisite helpers.

For example, phase membership of total constraints is checked while loading, again in `DomainSpec`, and again in `Case`. Symbol consistency and reaction species are also checked across multiple layers.

I would use this division:

## Keep validation at external boundaries

Keep:

* YAML unknown-key checks;
* rational parsing;
* species lookup;
* requested family lookup;
* family result is a mapping of `Reaction` objects;
* reaction IDs match family keys;
* symbol ownership;
* domain feasibility;
* positive temperature;
* valid total group phase;
* finite real rate constants;
* operation budgets.

These protect against realistic user or reaction-author errors.

## Remove internal defensive checks

Remove:

* `MappingProxyType` throughout;
* padded-whitespace checks on every internal identifier;
* Boolean checks on fields already created internally;
* `Role`/`blocking` agreement checks;
* `RunResult` ordering checks;
* “context belongs to this exact case” checks;
* repeated duplicate-ID checks after planning;
* runtime errors asserting that passing prerequisites contain all expected reactions;
* phase validation after the loader already constructed the phase group;
* repeated domain feasibility checks after construction;
* exception catches for impossible missing species after case validation;
* broad exception wrapping around family builders.

The frozen dataclasses plus a small amount of discipline are sufficient for this tool. Deeply immutable dictionaries are not providing meaningful safety.

## Do not remove the algorithmic safeguards

The operation limits and bounded witness searches are not defensive clutter. They prevent SymPy from running without bound. Keep:

```text
factorisation size limit
derivative fallback size limit
residual size limit
face result/search limit
witness count limit
```

Those are part of the algorithm’s termination contract, not enterprise-style validation.

---

# 8. Simplify the domain without weakening it

The unified domain is worth keeping. It is one of the strongest parts of the refactor.

But `domain.py` repeats exact conversion, reconstructs combined interval mappings, contains separate total-filling logic for witnesses and extrema, and carries broad interval restriction logic that most callers do not need.

Specific reductions:

### Centralize exact conversion

There are separate `_exact()` implementations in `domain.py` and `proof/analysis.py`.

Use one:

```python
exact_expr(value)
```

alongside `parse_rational()`.

### Store all intervals directly

Instead of:

```python
intervals
parameter_intervals

@property
def all_intervals(self):
    return {**self.intervals, **self.parameter_intervals}
```

store a single mapping plus a concentration-symbol set:

```python
all_intervals: dict[Symbol, Interval]
concentrations: frozenset[Symbol]
```

Then:

```python
parameter_symbols = all_intervals.keys() - concentrations
```

or retain T/P directly from the case.

### Share total-constraint filling

Both `exact_witness()` and `_extreme()`:

1. construct a box point;
2. calculate total deficit;
3. distribute concentration capacity.

Extract:

```python
satisfy_totals(point, costs=None)
```

The witness passes no costs; affine optimization passes coefficient costs.

### Make `restrict()` only narrow

Current `restrict()` clips values that lie outside the current interval and carefully combines open/closed flags.

All current callers intend to intersect the domain with a tighter interval. Use:

```python
lower = max(current.lower, requested_lower)
upper = min(current.upper, requested_upper)
```

and allow infeasibility to be detected normally.

### Use `Phase` in total constraints

Instead of:

```python
TotalConstraint(name="gas", ...)
```

then parsing that name back into `Phase` inside `Case`, use:

```python
TotalConstraint(phase=Phase.GAS, ...)
```

This removes a class of string validation and cross-object checking.

I would target **260–300 lines** for the domain while preserving open and closed interval endpoints, exact affine bounds, exact witnesses, and chamfer constraints.

---

# 9. Cut the loader nearly in half

`loading.py` is long because it provides custom, very specific error handling for every nested field and validates several things that are validated again by the model.

The loader should still reject unknown YAML keys, but it can become much more declarative.

## Table-drive total loading

Instead of two calls and phase-specific branching spread through `_load_total()`:

```python
TOTAL_MODES = {
    Phase.GAS: {"none", "explicit", "ideal_gas_minimum"},
    Phase.SOLID: {"none", "explicit"},
}
```

Then one compact function can:

1. obtain default phase species;
2. apply optional override;
3. resolve the minimum;
4. construct `TotalConstraint`.

## Stop validating the check plan while loading

`_load_checks()` currently calls `plan_checks()` to validate the configuration, then the runner plans it again later. Remove that call. Unknown check IDs can be detected when planning.

## Use typed configurations

A small object:

```python
@dataclass(frozen=True)
class RunConfig:
    profile: str = "physical"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
```

and:

```python
@dataclass(frozen=True)
class ReportConfig:
    format: str = "text"
    verbosity: str = "failures"
    output: Path | None = None
```

will simplify `Case`, `CLI`, and runner code more than the dataclasses cost.

## Let family exceptions propagate

This block is unnecessary:

```python
try:
    families[family_id] = _build_family(...)
except Exception as error:
    if isinstance(error, (ValueError, TypeError)):
        raise
    raise RuntimeError(...) from error
```

The original exception and traceback are more useful to the reaction developer.

## Retain dynamic family validation

The family module is external code. Keep these checks:

* `build_family` exists;
* it returns a nonempty mapping;
* values are `Reaction`;
* local and qualified IDs match.

That is a legitimate trust boundary.

A target of **240–280 lines** is realistic.

---

# 10. Merge small check modules by mathematical responsibility

The current layout creates a module, imports, runner function, result conversion and `__all__` for several checks that are only tens of lines of actual mathematics.

Suggested grouping:

```text
checks/
  core.py          # specs, profiles, planning, execution
  chemistry.py     # atoms and mass
  rates.py         # definedness, sign, Lipschitz, depletion
  invariance.py
  recovery.py
  analyses.py
```

## `chemistry.py`

Remove `AtomConservationResult` and `MassConservationResult` unless they are genuinely used as a public API.

Use:

```python
def atom_imbalance(reaction, species) -> dict[str, Rational]
def mass_imbalance(reaction, species) -> MassBalance
```

The current files spend considerable space on result dataclasses, mapping proxies and catches for species that the validated case cannot be missing.

## `rates.py`

Definedness, nonnegative rate, Lipschitz and zero at depletion all iterate over reactions, consult a prerequisite, invoke the analyzer, and convert a proof into a finding.

A shared private loop is appropriate here:

```python
def check_rates(context, dependency, analyze, describe):
    ...
```

Do not create a generic framework for every conceivable check; just remove the four copies of the same loop.

## `invariance.py`

Keep this separate because it expresses a theorem rather than symbolic analysis, but remove the defensive checks that reconstruct and verify prerequisite contents.

The DAG guarantees that:

* rate nonnegativity passed;
* depletion passed;
* network Lipschitz passed.

The current file spends a meaningful fraction of its length proving that the runner did what the runner says it did, and stores a detailed per-face evidence dictionary that can be reconstructed from the case.

The actual theorem can be implemented in roughly 50–80 lines.

---

# 11. Reduce the optional analyses

`analyses.py` is reasonably written, but it combines several outputs and a bounded recursive face search.

Reasonable reductions:

* remove `_linear_text`; let the reporter render a structured linear expression;
* store vectors and equations, not a human string plus the same information in evidence;
* make `_minimal_sets()` iterative using an explicit stack instead of a nested recursive `search()`;
* remove connected-component reporting if it is not used anywhere downstream;
* do not include full `required_supports` in every result unless verbose output explicitly requests it.

The recursive nested function is one place where removing nesting would genuinely improve readability:

```python
stack = list(seeds)
while stack:
    face = stack.pop()
    ...
```

That should reduce the module to around **250–300 lines**.

However, if you want the entire production package to reach a genuine 50% reduction, this is where I would make a product decision:

* keep conserved quantities;
* temporarily remove structural face search;
* temporarily remove steady-state equation reporting.

They are nonblocking analyses, not core physical validation. Removing them would save more than repeated style cleanups elsewhere.

---

# 12. Clean the reaction-family files

These files should remain readable rather than maximally compressed, but there is easy duplication.

## Xu–Froment

The file currently has:

* three nearly identical equilibrium-constant functions;
* six private rate functions;
* six public wrappers that reconstruct the common terms;
* repetitive reaction constructors;
* a nested partial-pressure function.

Use:

```python
EQUILIBRIUM_COEFFICIENTS = {
    "smr": (...),
    "wgs": (...),
    "overall": (...),
}

def equilibrium_constant(kind, temperature):
    return exp(polynomial(temperature, EQUILIBRIUM_COEFFICIENTS[kind]))
```

Delete the public `smr_fw_rate(symbols)` style wrappers unless something outside the family actually uses them. `build_family()` already computes the shared terms once.

A small reaction constructor helper can remove repeated IDs, catalysts and dictionary formatting.

## Medrano

The file has both `MedranoTerms` and `MedranoFamilyTerms`, plus a separate `MedranoReactionState`. It also maintains several parallel parameter dictionaries indexed by the same three component keys.

Use:

```python
@dataclass(frozen=True)
class ComponentParameters:
    cs: float
    r0: float
    k0: float
    activation_energy: float
    order: float
    d0: float
    diffusion_energy: float
    kx: float
    kxe: float
    b: float
```

with one mapping:

```python
PARAMETERS = {
    "H2": ComponentParameters(...),
    "CO": ComponentParameters(...),
    "O2": ComponentParameters(...),
}
```

Then:

* use one shared family-terms object;
* calculate `gas_fraction` locally;
* return `(conversion, unreacted)` rather than a dataclass;
* remove public `reduction_h2_rate()` wrappers if unused.

Also verify this section:

```python
c_power_kinetic = _gas_concentration_power_expr(...)
c_power_diffusive = _gas_concentration_power_expr(...)
```

The calls are currently identical. If that is intentional, calculate it once. If they were meant to differ, this is a latent kinetics error that the cleanup has exposed.

---

# 13. Simplify the species registry

The current registry defines:

* `PropertyRegistry`;
* a mapping proxy;
* key/record-ID validation;
* a `get_record()` wrapper;
* an `_species()` wrapper;
* the actual data table.

For this project, a plain dictionary is enough:

```python
SPECIES = {
    "CH4": Species(...),
    ...
}

def species(species_id):
    try:
        return SPECIES[species_id]
    except KeyError:
        raise ValueError(f"Unknown species {species_id!r}")
```

The dictionary key and `Species.id` are written next to one another in the same file. A runtime class ensuring they agree is not offering much.

---

# 14. Reduce tests and remove the refactor handoff file

The tests are not production code, but they now account for about 1,600 lines. The suite contains useful mathematical tests, but there is a lot of repeated YAML construction and testing of internal framework safeguards. A representative file repeatedly writes complete case YAML strings and tests architecture details such as how often a helper was called.

I would retain approximately **700–900 lines** of tests:

```text
test_model_and_domain.py
test_expression_proofs.py
test_physical_checks.py
test_augmented_recovery.py
test_cases.py
```

Use:

* shared case/YAML builders;
* parametrized expression theorem cases;
* one integration test for each bundled case;
* only a few DAG tests;
* no tests for deleted `MappingProxyType`, result-validation or evidence-shape machinery.

Also remove `refactor.md` once the actual documentation is written. It is about 58 KB and was introduced as a 2,200-line handoff document, so it contributes heavily to the apparent repository size even though it is not executable code.

---

# A smaller target structure

I would aim for:

```text
src/rxn_checker/
  __init__.py
  model.py              # Species, symbols, Reaction, Case, configuration
  domain.py
  loading.py
  context.py            # Includes network construction

  proof/
    analysis.py
    lipschitz.py

  checks/
    core.py             # Check definitions, profiles, planning, execution
    chemistry.py
    rates.py
    invariance.py
    recovery.py
    analyses.py

  reactions/
    ...
  species.py
  reporting.py
  cli.py
```

That reduces the core package from roughly three dozen Python modules to the low twenties.

Merging files by itself is not the goal. The useful aspect is that it makes duplicated responsibilities visible:

* `model.py` owns internal validity;
* `loading.py` owns external schema parsing;
* `analysis.py` owns all sign/zero/witness logic;
* `core.py` owns all DAG execution;
* checks contain only their theorem-specific logic;
* `reporting.py` owns formatting.

---

# Recommended reduction sequence

## Pass 1: Results and reporting

* Flatten `Evidence` into `Finding.details`.
* Remove `Role`.
* Remove `MappingProxyType`.
* Remove special evidence-kind handling from the text renderer.
* Add generic dataclass/SymPy JSON serialization.

This is low mathematical risk and will immediately shorten nearly every check.

**Expected saving: 250–350 lines.**

## Pass 2: Check framework

* Define profiles using terminal checks.
* Remove profile membership from `CheckSpec`.
* Remove custom registries.
* Remove repeated registry validation.
* Fix stage-failure behaviour instead of configuring it.
* Merge definitions, registry, runner and prerequisite helper.

**Expected saving: 250–350 lines.**

## Pass 3: Generic proof operations

* Collapse proof result types.
* Remove active-variable propagation outside Lipschitz.
* Centralize exact conversion and candidate generation.
* Add `prove_zero()` and `prove_sum()`.
* Use small internal Lipschitz results.

**Expected saving: 450–650 lines.**

## Pass 4: Rewrite thin checks

* Rewrite zero depletion on `prove_zero()`.
* Rewrite negative-side recovery on `prove_sum()`.
* Merge physical rate check wrappers.
* Simplify invariance to theorem composition.
* Merge atom and mass conservation.

**Expected saving: 500–700 lines.**

## Pass 5: Trust-boundary cleanup

* Merge `Case` into `model.py`.
* Remove duplicated model/domain/loader validation.
* Simplify domain total handling.
* Merge network into context.
* Simplify species registry.

**Expected saving: 300–450 lines.**

## Pass 6: Reaction and analysis cleanup

* Remove unused public rate wrappers.
* Consolidate family parameter tables.
* Simplify optional analyses.
* Remove repeated human-text-plus-structured-data construction.

**Expected saving: 250–400 lines.**

## Pass 7: Tests and repository cleanup

* Parameterize the theorem tests.
* Add shared YAML fixtures.
* Delete tests of removed safeguards.
* Remove `refactor.md` after the actual documentation replaces it.

This does not reduce production LoC, but it should roughly halve the overall visible repository size.

---

# What I would not cut

To hit a number, I would **not** remove:

* exact rational parsing;
* physical and augmented domains;
* open versus closed interval semantics;
* total concentration constraints;
* exact affine optimization;
* expression definedness;
* Lipschitz constants;
* exact failure witnesses;
* operation budgets;
* symbol ownership checks;
* selected-family validation;
* YAML unknown-key checks;
* a compact local theorem test suite.

Those are either mathematically essential or protect exactly the kinds of errors this repository exists to find.

---

# Realistic target

My recommended target would be:

```text
Production Python:
    approximately 3,800–4,200 lines

Tests:
    approximately 700–900 lines

Documentation:
    README + concise design notes
    no 2,200-line temporary refactor plan
```

That is probably a **35–40% reduction** from the present production package while retaining the current mathematical functionality.

A stretch target around **3,200–3,500 production lines** is possible, but I would only pursue it by making at least one explicit scope cut:

* remove structural face search;
* remove steady-state equation analysis;
* remove derivative fallback;
* greatly simplify structured JSON evidence;
* stop exposing the low-level model and proof internals as a public API.

I would not target 50% through indiscriminate shortening. That tends to replace obvious code with dense generic code, which would make this particular project worse.

The most important rule for the next pass is:

> **Do not create a new abstraction unless it deletes at least two existing implementations.**

That means `prove_zero()` and `prove_sum()` are worthwhile. Another generic check framework is not.
