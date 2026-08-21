# Implementation Plan: DAETools-Oriented Evaluation Profile

## 1. Purpose

Add a nonblocking `evaluation_profile` analysis to `rxn-checker` that estimates the computational burden of reaction kinetics when equivalent expressions are implemented in DAETools.

The analysis remains **entirely SymPy-based**. It does not import DAETools, execute the DAETools model, or benchmark SymPy. Its output describes:

- mathematical work per cell and per residual evaluation;
- expensive operation types;
- expression-tree size and depth;
- structural automatic-differentiation burden;
- repeated calculations within and between rates;
- source-assembly and residual-Jacobian sparsity;
- compatibility with the mathematical operations exposed by the DAETools kinetics layer.

The result is a comparative profile, not a prediction in seconds.

---

## 2. Model the actual DAETools use case

### 2.1 Per-cell cost

`multisolid-CL` constructs reaction-rate equations over the cell-centre domain and gives each reaction a separate `R_rxn` variable. Species source expressions then refer to those rate variables through the stoichiometric coefficients. 

Consequently, the profile should describe:

```text
cost per cell per residual evaluation
```

It should not multiply by an assumed number of cells or solver iterations. Those belong to the eventual simulation configuration and solver behaviour.

### 2.2 Count mathematical operations, not Python construction

Do not count:

- calls to Python helper functions;
- dataclass construction;
- dictionary lookups;
- `Constant(...)` wrappers;
- unit-expression construction;
- SymPy evaluation time.

Those operations happen while the DAETools equation tree is being constructed, rather than whenever the solver evaluates the residual.

Count the mathematical operations represented by the final SymPy expression:

- arithmetic;
- powers;
- exponentials and logarithms;
- square roots;
- clamps and absolute values;
- branches or unsupported functions.

### 2.3 Include temperature and pressure as DAE dependencies

In `rxn-checker`, temperature and pressure are called parameter symbols because the concentration-domain analysis treats them as uniformly bounded parameters. In `multisolid-CL`, however, both `T` and `P` are DAETools variables. 

Therefore:

```python
dae_symbols = context.case.symbols.all_symbols
```

must be used for dependency and Jacobian-work calculations, not just `concentration_symbols`.

The report should still split dependencies into:

```text
concentrations
temperature/pressure
all DAE inputs
```

---

## 3. Report two representations of the kinetics

There is currently an important structural difference between the repositories:

- `rxn-checker` represents Xu–Froment forward and backward directions as six separate nonnegative reactions. 
- `multisolid-CL` implements three signed net rates, each containing a forward-minus-backward driving force. 

Profiling only the declared `Reaction.rate` expressions would therefore misrepresent the actual DAETools implementation.

The analysis should report both views.

### 3.1 Declared-rate view

Profile every `Reaction.rate` exactly as declared.

This remains useful because it identifies which directional expression is expensive and preserves the existing reaction-checker semantics.

### 3.2 Source-equivalent flux view

Group stoichiometrically collinear reaction columns. If

\[
\nu_j=\alpha_j\bar{\nu},
\]

replace their source contribution

\[
\sum_j \nu_j r_j
\]

with

\[
\bar{\nu}\left(\sum_j\alpha_jr_j\right).
\]

For a reversible pair this produces

\[
r_{\mathrm{net}}=r_{\mathrm{fw}}-r_{\mathrm{bw}}.
\]

The result is an exact, fully symbolic, source-equivalent implementation view. It should turn the current reforming case into approximately:

```text
9 declared directional rates
6 source-equivalent flux expressions
```

namely three Medrano fluxes and three net Xu–Froment fluxes.

This view should be labelled clearly:

```text
source-equivalent flux profile
```

It is not a claim that DAETools automatically performs this grouping.

### 3.3 Reuse the existing primitive-vector logic

The code already normalises rational vectors to primitive integer vectors when reporting conserved quantities. Move that small helper from `checks/analyses.py` into `network.py` under a generic name such as:

```python
primitive_integer_vector(vector)
```

Use it both for conserved quantities and for grouping proportional stoichiometric columns. This avoids implementing the same rational scaling logic twice.

---

## 4. Operation taxonomy

Use a small fixed operation vocabulary that closely matches the DAETools kinetics wrapper.

The current wrapper exposes arithmetic together with `Exp`, `Log`, `Max`, `Min`, `Abs`, and `Sqrt`. 

```python
OPERATION_ORDER = (
    "add",
    "multiply",
    "reciprocal",
    "integer_power",
    "general_power",
    "sqrt",
    "exp",
    "log",
    "abs",
    "min",
    "max",
    "piecewise",
    "other_function",
)
```

### Counting rules

| SymPy expression | Profile operation |
|---|---|
| `Add(a, b, c)` | 2 additions |
| `Mul(a, b, c)` | 2 multiplications |
| `Pow(x, -1)` | 1 reciprocal |
| `Pow(x, 1/2)` | 1 square root |
| `Pow(x, n)`, integer \(n\neq -1\) | 1 integer power |
| Other `Pow` | 1 general power |
| `exp(x)` | 1 exponential |
| `log(x)` | 1 logarithm |
| `Abs(x)` | 1 absolute value |
| `Min(a, b, c)` | 2 minimum selections |
| `Max(a, b, c)` | 2 maximum selections |
| `Piecewise` | 1 piecewise operation plus branch count |
| Unknown applied function | 1 other function and compatibility warning |

Do not assign arbitrary default weights. An exponential is not converted into “20 multiplications,” because the relative cost depends on the DAETools evaluator, compiler, processor, and derivative mode.

The operation histogram itself is the primary result.

For convenient ranking, expose only these derived values:

```python
total_operations
transcendental_operations
switch_operations
```

where:

```text
transcendental = exp + log + sqrt + general_power
switch = abs + min + max + piecewise
```

These are transparent sums rather than pretend timing models.

---

## 5. Core expression statistics

For each output expression, calculate:

```text
operation histogram
total operations
tree nodes
unique subexpressions
maximum depth
concentration dependencies
temperature/pressure dependencies
all DAE dependencies
structural Jacobian entries
dependency-weighted AD work
unsupported DAETools operations
```

### 5.1 Tree and DAG size

Report both:

- `tree_nodes`: repeated occurrences counted repeatedly;
- `unique_nodes`: structurally equal SymPy subexpressions counted once.

Their difference is an inexpensive first indication of reuse opportunity.

### 5.2 Structural Jacobian entries

For rate \(r_j\), define:

\[
d_j=
\left|
\operatorname{free}(r_j)
\cap
\text{DAE symbols}
\right|.
\]

This is the number of structurally nonzero derivatives of that rate with respect to its direct DAE inputs.

Do not calculate explicit symbolic derivatives for this analysis. DAETools differentiates its own expression tree, and the expanded SymPy derivative is neither the expression DAETools evaluates nor a cheap object to construct.

### 5.3 Dependency-weighted AD work

Use a simple proxy for automatic-differentiation burden.

For each operation node \(q\), determine the set of DAE symbols on which it depends:

\[
D(q)=
\bigcup_{a\in\operatorname{children}(q)}D(a).
\]

Then accumulate:

\[
W_{\mathrm{AD}}
=
\sum_q
m(q)\,|D(q)|,
\]

where \(m(q)\) is the operation multiplicity, such as \(n-1\) for an \(n\)-argument addition.

This measures how much derivative information must propagate through the expression without constructing derivatives explicitly.

Call it:

```text
dependency-weighted AD work
```

not “Jacobian FLOPs” or “Jacobian evaluation time.” It is a comparative structural metric.

---

## 6. Common-subexpression analysis

Provide three cost levels.

### 6.1 Raw declared cost

Sum the independently evaluated declared rates:

```text
raw declared-rate cost
```

This is the conservative estimate when no computational sharing between rate equations is assumed.

### 6.2 Per-output CSE

Run `sympy.cse` independently on each rate or net-flux expression.

This identifies duplicated work inside one output. It is particularly relevant to the current Medrano mirror, where the same gas-concentration-power expression is constructed separately as `c_power_kinetic` and `c_power_diffusive`. 

Report:

```text
raw operations
operations after local CSE
local saving
temporary count
peak live temporaries
```

### 6.3 Global CSE

Run CSE over all declared rates together, and separately over all source-equivalent fluxes:

```python
replacements, outputs = sp.cse(
    expressions,
    symbols=sp.numbered_symbols("t"),
    order="canonical",
)
```

Do not use `simplify`, `expand`, `factor`, or the optional aggressive CSE optimisations. The purpose is to detect exact repeated work, not to launch an unbounded algebraic optimisation pass.

Report:

```text
raw total
CSE total
shareable operation reduction
temporary count
peak live temporaries
```

The global-CSE result is an **opportunity estimate**, not an assumption about DAETools execution. Python helper variables and identical DAETools subtrees must not automatically be treated as runtime memoisation.

The text summary should therefore use wording such as:

```text
38% of the declared mathematical work is structurally shareable.
```

not:

```text
The optimized implementation will be 38% faster.
```

### 6.4 Shared-term recommendations

Independently enumerate repeated non-atomic subexpressions from the original expression trees.

For each candidate, record:

```text
expression
operation count
number of occurrences
reactions using it
within-rate or cross-rate reuse
individual potential saving
```

Calculate the individual saving as:

\[
(\text{occurrences}-1)
\times
\text{operations in the subexpression}.
\]

These savings are not additive because candidates can overlap.

Only show the largest five in text. Preserve the complete list in JSON evidence.

Likely Xu–Froment recommendations should include:

- temperature conversion;
- pressure conversion;
- partial pressures;
- hydrogen inverse pressure;
- adsorption constants;
- adsorption denominator;
- Arrhenius factors.

The current DAETools implementation already groups many of these into `XuFromentTerms`, which is good for readability, but each of the three rate hooks currently constructs its own terms. 

---

## 7. Peak live temporaries

CSE can reduce operations while creating a large number of intermediate values.

Implement a small liveness calculation:

1. CSE replacements are already topologically ordered.
2. Find the last replacement or output that uses each temporary.
3. Scan the replacement sequence.
4. Add a temporary after its definition.
5. Remove it after its final use.
6. Record the largest live set.

Report:

```text
temporary count
peak live temporaries
```

This is enough to expose pathological CSE results without building a compiler-like intermediate representation.

Do not attempt register allocation or memory-size prediction.

---

## 8. Kinetics-block residual structure

Because `multisolid-CL` represents rates as separate `R_rxn` variables, the profile should describe the actual residual block rather than only \(S\,r(c)\).

For \(m\) rate outputs:

### Rate equations

Each equation has the form

\[
R_j-r_j(c,T,P)=0.
\]

Its structural Jacobian contains:

- one entry for \(R_j\);
- one entry for each DAE symbol used by \(r_j\).

Therefore:

\[
N_{\mathrm{rate}}
=
m+\sum_j d_j.
\]

### Source coupling

Species equations refer to `R_rxn` through the nonzero stoichiometric coefficients. The additional structural coupling is:

\[
N_{\mathrm{source}}=\operatorname{nnz}(S).
\]

Report:

```text
rate equations
rate-input structural entries
stoichiometric source links
total kinetics-block structural entries
maximum rate dependency width
```

Calculate this for both:

- declared-rate layout;
- source-equivalent flux layout.

This is more representative of the DAETools model than explicitly forming and differentiating \(S\,r(c)\).

---

## 9. DAETools mirror compatibility

The profile should validate that each SymPy expression can be mirrored using the current `multisolid-CL` kinetics operation surface.

Supported directly:

```text
Add, Mul, numeric Pow
exp, log, sqrt
Abs, Min, Max
```

For every unsupported function, record:

```text
function name
decisive subexpression
affected outputs
```

Examples include:

- trigonometric functions not exposed by the current wrapper;
- symbolic exponents;
- `Piecewise`;
- custom undefined SymPy functions.

An unsupported function makes the affected profile `UNKNOWN`, but the rest of the statistics should still be returned.

This remains a nonblocking analysis; it should never produce a chemistry-level failure.

---

## 10. Minimal data model

Use four small immutable dataclasses. Do not build a hierarchy of expression-node classes or backend visitors.

```python
@dataclass(frozen=True)
class ExpressionStats:
    operations: tuple[tuple[str, int], ...]
    tree_nodes: int
    unique_nodes: int
    depth: int
    concentration_dependencies: tuple[sp.Symbol, ...]
    operating_dependencies: tuple[sp.Symbol, ...]
    ad_work: int
    unsupported_functions: tuple[str, ...]

    @property
    def total_operations(self) -> int:
        return sum(value for _, value in self.operations)


@dataclass(frozen=True)
class CSEStats:
    operations: tuple[tuple[str, int], ...]
    temporary_count: int
    peak_live_temporaries: int

    @property
    def total_operations(self) -> int:
        return sum(value for _, value in self.operations)


@dataclass(frozen=True)
class SharedTerm:
    expression: sp.Expr
    operations: int
    occurrences: int
    outputs: tuple[str, ...]
    estimated_saved_operations: int


@dataclass(frozen=True)
class EvaluationProfile:
    declared_outputs: tuple[tuple[str, ExpressionStats], ...]
    flux_outputs: tuple[tuple[str, ExpressionStats], ...]
    declared_cse: CSEStats
    flux_cse: CSEStats
    declared_local_cse: tuple[tuple[str, CSEStats], ...]
    flux_groups: tuple[Mapping[str, object], ...]
    shared_terms: tuple[SharedTerm, ...]
    declared_source_nnz: int
    flux_source_nnz: int
```

Use tuples instead of mutable dictionaries inside the proof layer so results are deterministic and hashable. Convert them to ordinary mappings only in the check adapter.

---

## 11. File layout

Keep the implementation contained.

```text
src/rxn_checker/
├── proof/
│   ├── evaluation.py              # pure operation/CSE/profile logic
│   └── __init__.py                # public exports
├── checks/
│   ├── evaluation_profile.py      # AnalysisContext → Finding adapter
│   └── core.py                    # registry entry/profile selection
└── network.py                     # shared primitive-vector normalisation

tests/
└── test_evaluation_profile.py
```

Suggested size target:

```text
proof/evaluation.py          220–300 lines
checks/evaluation_profile.py  50–80 lines
tests                         200–300 lines
```

A few hundred straightforward lines are preferable to splitting this into an operation visitor, cost model, CSE model, backend model, and report model.

---

## 12. Core API

Expose one public proof-layer function:

```python
def profile_evaluation(
    reactions: Sequence[Reaction],
    stoichiometry: sp.MatrixBase,
    concentration_symbols: Iterable[sp.Symbol],
    operating_symbols: Iterable[sp.Symbol],
) -> EvaluationProfile:
    ...
```

Everything else in `proof/evaluation.py` should be private.

The check runner becomes approximately:

```python
def run(context, _dependencies):
    profile = profile_evaluation(
        context.case.reactions,
        context.stoichiometry,
        context.case.symbols.concentration_symbols,
        context.case.symbols.parameter_symbols,
    )
    return Finding(
        context.case.name,
        Verdict.UNKNOWN if profile_has_unsupported_functions(profile)
        else Verdict.PASS,
        _summary(profile),
        Evidence("evaluation_profile", _evidence(profile)),
    )
```

Do not add evaluation logic to `ExpressionAnalyzer`. Its current responsibility is domain-based bounding and proof caching. Evaluation profiling does not use a domain and is structurally different from sign, definedness, and Lipschitz proofs.

---

## 13. Check registration

Register:

```python
_spec(
    "evaluation_profile",
    "Evaluation profile",
    Stage.ANALYSIS,
    CheckScope.CASE,
    (),
    run_evaluation_profile,
    blocking=False,
)
```

It should have no check dependencies. An expression can still be structurally profiled when atom conservation, definedness, or positivity fails.

Include it in:

```text
analysis
all
```

No reporting changes should be necessary. The existing result model accepts arbitrary structured evidence, and the JSON renderer already recursively handles mappings, sequences, enums, and SymPy expressions.  

---

## 14. Text report

Keep the text output compact. Put complete information in JSON.

Example:

```text
PASS     Evaluation profile
         9 declared rates: 1,842 operations/cell; 71 transcendental,
         46 switch operations. Global CSE leaves 1,163 operations
         with 24 temporaries, at most 11 live.

         6 source-equivalent fluxes: 1,391 raw operations; global CSE
         leaves 947. Kinetics residual structure: 6 rate equations,
         53 rate-input links, 22 stoichiometric links.

         Most expensive outputs: xu_froment.smr_net (312),
         medrano.reduction_co (247), medrano.reduction_h2 (239).

         Largest reuse opportunities: Xu-Froment adsorption denominator,
         total gas concentration, Medrano concentration-power term.
```

The exact values above are illustrative.

The summary should include:

- declared and source-equivalent output counts;
- raw and CSE operation totals;
- transcendental and switch counts;
- temporary count and peak liveness;
- kinetics-block structural entries;
- three most expensive outputs;
- unsupported-function count.

---

## 15. JSON evidence

Use one evidence object:

```json
{
  "target": "DAETools mathematical expression tree",
  "units": "operations per cell per residual evaluation",
  "declared": {
    "output_count": 9,
    "raw": {},
    "global_cse": {},
    "source_nnz": 0,
    "residual_jacobian_nnz": 0
  },
  "source_equivalent": {
    "output_count": 6,
    "groups": [],
    "raw": {},
    "global_cse": {},
    "source_nnz": 0,
    "residual_jacobian_nnz": 0
  },
  "outputs": {
    "reaction_or_flux_id": {
      "raw": {},
      "local_cse": {},
      "dependencies": {},
      "ad_work": 0,
      "unsupported_functions": []
    }
  },
  "shared_terms": []
}
```

Do not serialize CSE temporary symbols and every replacement by default. That would make the report large and difficult to use. Preserve only:

- operation totals;
- temporary counts;
- peak liveness;
- top repeated expressions.

---

## 16. Tests

### 16.1 Operation classification

Test exact counts for small expressions:

```python
a + b + c
a * b * c
a / b
sp.sqrt(a)
a**3
a**sp.Rational(3, 5)
sp.exp(a)
sp.Max(a, 0)
```

### 16.2 Dependency propagation

Verify:

```python
r = exp(-E / temperature) * A
```

reports:

- concentration dependency: `A`;
- operating dependency: `temperature`;
- two DAE dependencies;
- correct dependency-weighted AD work.

### 16.3 Local CSE

Use:

```python
q = a + b
expression = q**2 + exp(q)
```

and confirm local CSE extracts one temporary and reduces repeated work.

### 16.4 Cross-output CSE

Use:

```python
rates = (
    exp(-1 / T) * A,
    exp(-1 / T) * B,
)
```

and confirm global CSE shares the Arrhenius term while independent local CSE does not.

### 16.5 Stoichiometric grouping

Test:

```text
A → B
B → A
```

and confirm one source-equivalent flux:

\[
r_{\mathrm{fw}}-r_{\mathrm{bw}}.
\]

Also test:

- identical parallel stoichiometric directions;
- rationally proportional columns;
- unrelated columns;
- zero coefficients;
- deterministic group ordering.

### 16.6 Residual structure

For a small network, verify:

\[
N_{\mathrm{rate}}
=
m+\sum_jd_j
\]

and:

\[
N_{\mathrm{source}}=\operatorname{nnz}(S).
\]

### 16.7 Unsupported functions

Use a custom function or `Piecewise` and confirm:

- useful partial statistics remain;
- the affected output is marked unsupported;
- the check verdict is `UNKNOWN`;
- the analysis remains nonblocking.

### 16.8 Reforming regression

For the current reforming case, avoid brittle assertions on every exact count. Assert stable structural properties:

```text
9 declared rates
6 source-equivalent fluxes
global CSE does not increase operation count
at least one cross-rate shared term
nonzero exponential count
nonzero general-power count
nonzero switch count
no unsupported current mirror functions
temperature and pressure included as DAE dependencies
```

Also assert that the repeated Medrano gas-concentration-power expression appears as a reuse opportunity.

---

## 17. Implementation sequence

### Step 1 — Pure operation traversal

Implement:

```text
operation classification
tree size
unique-node count
depth
dependency propagation
AD-work proxy
unsupported-function collection
```

Test before adding any checker integration.

### Step 2 — CSE analysis

Add:

```text
local CSE
global CSE
operation totals after CSE
temporary count
peak temporary liveness
shared-term ranking
```

### Step 3 — Source-equivalent fluxes

Move the primitive-vector helper, group proportional stoichiometric columns, and construct exact net-flux expressions without expanding them.

### Step 4 — DAETools residual structure

Add rate-equation dependency entries and stoichiometric source links for both representations.

### Step 5 — Check and report integration

Add the case-scoped nonblocking analysis, concise text summary, complete JSON evidence, and registry/profile entries.

### Step 6 — Validate against `multisolid-CL`

Compare the highest-cost and highest-reuse findings manually against:

```text
packed_bed/kinetics/reforming_xu_froment.py
packed_bed/kinetics/nickel_medrano.py
```

The expected findings should correspond to recognisable implementation structures rather than arbitrary SymPy internals.

---

## 18. Deliberate non-goals

Do not include the following in this feature:

- SymPy wall-clock benchmarks;
- DAETools imports;
- automatic DAETools code generation;
- automatic rewriting of reaction expressions;
- exact solver-step predictions;
- explicit symbolic Jacobian construction;
- symbolic Hessians;
- arbitrary “good/bad” operation thresholds;
- one opaque solver-friendliness score;
- automatic creation of auxiliary DAETools variables from CSE;
- claims that theoretical CSE savings translate directly into runtime savings.

Those either belong in the later differential solver profile or require measurement inside an actual DAETools simulation.

---

## 19. Completion criteria

The feature is complete when it can answer, for every case:

1. Which rate or net flux contains the most mathematical work?
2. Which expensive operations dominate it?
3. How many DAETools variables does it structurally depend on?
4. What is its comparative automatic-differentiation burden?
5. How much repeated work exists inside each output?
6. How much work is shared across outputs?
7. Which exact subexpressions are worth extracting or reusing?
8. How many rate/source Jacobian links does the kinetics block introduce?
9. Can every SymPy construct be mirrored by the current DAETools kinetics operation layer?
10. How does the declared directional representation compare with the source-equivalent net-rate representation?

That is enough to guide reaction-expression refactoring without turning `rxn-checker` into a compiler or a DAETools benchmarking framework.