# Reaction Checker Rewrite — Agent Handoff Specification

Your corrections change several of my earlier recommendations in useful ways. The rewrite should **not** introduce a vocabulary of reaction-specific primitives, a unit system, reversible-reaction machinery, or a large software-engineering apparatus. It should concentrate on a small number of mathematically clear abstractions:

1. one exact reaction-system model;
2. one unified domain model producing physical and augmented domains;
3. one reusable symbolic-analysis engine for general SymPy expressions;
4. one explicit DAG of checks;
5. one run context that computes every shared object once;
6. concise, structured reporting.

The intended result is a smaller and more understandable research tool, not an extensible enterprise framework.

---

# 1. Non-negotiable design decisions

The implementing agent should treat the following as fixed requirements.

| Decision                  | Requirement                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| Reaction expressions      | Arbitrary ordinary SymPy expressions                                                     |
| Custom kinetic primitives | Do not require them and do not make correctness depend on them                           |
| Reaction direction        | Every `Reaction` is one directional rate law                                             |
| Reversible abstraction    | Never add a `ReversibleReaction` type or forward/reverse pairing logic                   |
| Units                     | Strict SI-only convention; no unit-validation system                                     |
| Stoichiometry             | Exact nonnegative rational coefficients                                                  |
| Elemental composition     | Exact positive rational atom counts, including non-integral compositions such as Fe₀.₉₄O |
| Concentration domains     | Independent box or phase-chamfered box                                                   |
| Domain variants           | Physical and augmented, generated from one specification                                 |
| Temperature and pressure  | Bounded parameters, not Lipschitz state coordinates                                      |
| Lipschitz claim           | Lipschitz on an open neighbourhood of the selected domain                                |
| Lipschitz output          | Return a certified constant bound, not necessarily the minimal constant                  |
| Proof style               | Symbolic and exact; no numerical sampling presented as proof                             |
| Check execution           | Explicit dependency DAG with shared cached analyses                                      |
| Failure flow              | Basic chemistry failures terminate later stages                                          |
| Testing                   | Small local mathematical regression suite; no GitHub Actions or other CI machinery       |

Two opposing directional reactions may coexist in a network, as they currently do in Xu–Froment, but the checker must never infer that they form a reversible pair or apply special reversible-reaction semantics.

---

# 2. Central interpretation of the domain

The domain should be a **robust admissible envelope**, not an exact thermodynamic state manifold.

In particular, the ideal-gas option should be used to calculate a safe minimum total gas concentration,

[
c_{\mathrm{gas,min}}
====================

\frac{P_{\min}}{R T_{\max}},
]

rather than imposing the exact nonlinear equality

[
P=RT\sum_i c_i
]

inside every proof.

This is the right interpretation for four reasons:

1. It produces the requested chamfered box.
2. It remains valid when a solver error makes individual concentrations inconsistent with the exact ideal-gas closure.
3. It keeps the domain linear and allows exact, fast bound calculations.
4. It makes a counterexample inside the declared domain a genuine counterexample, rather than a point that might be rejected later for violating an unmodelled equality.

Temperature and pressure remain bounded parameters. They can appear in rate expressions and affect uniform bounds, but the Lipschitz distance is measured only in concentration space.

---

# 3. Mathematical model

## 3.1 Symbols

Replace the current conceptual model, where temperature and pressure are mixed into `StateVariables`, with:

```python
@dataclass(frozen=True)
class CaseSymbols:
    concentrations: Mapping[str, sympy.Symbol]
    temperature: sympy.Symbol
    pressure: sympy.Symbol
```

The distinction is semantic:

* concentrations are ODE state coordinates;
* temperature and pressure are externally ranged parameters;
* all symbolic checks must hold uniformly over the configured temperature and pressure intervals.

Reaction builders may still use:

```python
symbols.concentration("CH4")
symbols.temperature
symbols.pressure
```

The change is in analysis semantics, not necessarily in builder ergonomics.

This also makes the Lipschitz constant well-defined in a physically coherent coordinate space: all active coordinates are concentrations and therefore have the same SI dimension.

## 3.2 Species

Use an immutable species definition approximately like:

```python
@dataclass(frozen=True)
class Species:
    id: str
    name: str
    phase: Phase
    atoms: Mapping[str, sympy.Rational]
    molar_mass: sympy.Rational | None
```

`Phase` should initially contain only:

```python
class Phase(StrEnum):
    GAS = "gas"
    SOLID = "solid"
```

Atom counts must be positive rationals, not integers:

```python
Species(
    id="Fe0.94O",
    name="Wüstite",
    phase=Phase.SOLID,
    atoms={"Fe": Rational(47, 50), "O": Rational(1)},
    molar_mass=...,
)
```

Acceptable input forms for a rational number should include:

```text
1
0.5
"1/2"
"0.94"
```

All should be converted using the decimal spelling:

```python
sympy.Rational(str(value))
```

Do not convert through binary floating-point arithmetic after parsing.

## 3.3 Reactions

Use exact rational coefficients:

```python
@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: Mapping[str, Rational]
    products: Mapping[str, Rational]
    catalysts: tuple[str, ...]
    rate: Expr
```

Required validation:

* reaction ID is nonempty and unique;
* all side coefficients are finite, positive rationals;
* atom counts may be fractional;
* catalysts are unique;
* catalysts are not reactants or products;
* at least one side is nonempty;
* all species exist in the case;
* all free symbols belong to the case;
* the expression contains no explicit `nan`, `zoo`, infinity, or complex constants;
* the net stoichiometric vector is not identically zero.

`net_stoichiometry` should be computed once and stored or cached as exact rationals.

Do not use `float` for reaction stoichiometry. The current code accepts floating coefficients and later reconstructs rationals from their strings. The rewritten model should be exact from the start.

## 3.4 Inerts

An inert species means only:

* it has a zero stoichiometric row;
* it cannot be a reactant, product, or catalyst;
* it may appear in a rate expression through dilution or a total concentration.

Do not make inert concentrations strictly positive as a special domain rule.

In the physical domain an inert is nonnegative like every other concentration. In the augmented domain it may become negative if an excursion is configured. Since its source term is zero, the recovery analysis should naturally conclude:

* non-repelling: yes, because (F_i=0);
* strictly attracting: no;
* behaviour: stuck until the numerical method corrects it.

That is a useful result rather than a condition to hide through special bounds.

---

# 4. Unified domain model

## 4.1 One specification, two generated domains

The case should contain one `DomainSpec`. From it, construct:

```python
physical_domain = domain_spec.build(DomainKind.PHYSICAL)
augmented_domain = domain_spec.build(DomainKind.AUGMENTED)
```

The physical and augmented domains must never be configured independently. Otherwise they will eventually drift and checks will use subtly different assumptions.

## 4.2 Domain geometries

Support exactly two concentration geometries in the initial rewrite.

### Independent domain

Every concentration varies independently inside its own interval.

Physical:

[
0 \le c_i \le u_i.
]

Augmented:

[
\ell_i^{\mathrm{exc}} \le c_i \le u_i,
]

where (\ell_i^{\mathrm{exc}}\le0). If a species has no configured excursion, use zero.

There are no total-concentration constraints.

### Chamfered domain

Use the same individual bounds, plus one optional gas-total constraint and one optional solid-total constraint:

[
\sum_{i\in G} c_i \ge C_{G,\min},
\qquad
\sum_{i\in S} c_i \ge C_{S,\min}.
]

The total constraints are identical in the physical and augmented variants. Individual concentrations may become negative in the augmented domain, but the selected total remains safely positive.

Each group should allow an optional explicit species list. The default list is all species of the relevant phase. This matters for solids: a user may want the protected solid inventory to be `Ni + NiO`, excluding an inert support.

Do not introduce arbitrary overlapping concentration groups in the first rewrite. One gas group and one solid group are enough.

## 4.3 Total minimum modes

For gas:

```text
none
explicit
ideal_gas_minimum
```

For solids:

```text
none
explicit
```

`ideal_gas_minimum` means:

[
C_{G,\min}
==========

\frac{P_{\min}}{R T_{\max}}.
]

It does **not** impose an exact ideal-gas equality.

## 4.4 Temperature and pressure box

Temperature and pressure should use:

[
T_{\min}\le T\le T_{\max},
\qquad
P_{\min}\le P\le P_{\max}.
]

Both ranges are used in expression bounds. They are unchanged between physical and augmented concentration domains.

Require:

* (T_{\min}>0);
* (P_{\min}>0) when the gas minimum is derived from the ideal gas law;
* finite, correctly ordered bounds.

## 4.5 Domain representation

The core object should contain structured bounds rather than a loose bag of SymPy inequalities:

```python
@dataclass(frozen=True)
class ConcentrationDomain:
    kind: DomainKind
    intervals: Mapping[Symbol, Interval]
    parameter_intervals: Mapping[Symbol, Interval]
    total_constraints: tuple[TotalConstraint, ...]
```

Suggested supporting objects:

```python
@dataclass(frozen=True)
class Interval:
    lower: Expr
    upper: Expr
    lower_closed: bool = True
    upper_closed: bool = True


@dataclass(frozen=True)
class TotalConstraint:
    name: str
    symbols: tuple[Symbol, ...]
    minimum: Expr
```

The domain should expose a small API:

```python
domain.interval(symbol)
domain.restrict(symbol, lower=..., upper=..., strict_lower=False, strict_upper=False)
domain.affine_bounds(expression)
domain.is_feasible()
domain.exact_witness(preferences=...)
```

Do not make general relational constraint manipulation the public abstraction.

## 4.6 Exact feasibility

For the requested domains, feasibility is simple.

For each interval:

[
\ell_i\le u_i.
]

For each total constraint:

[
\sum_i u_i\ge C_{\min}.
]

Because gas and solid groups are disjoint, no general linear-programming package is needed merely to check domain feasibility.

---

# 5. Exact affine bounds without a general symbolic LP

A large amount of the current complexity comes from treating a simple box-plus-chamfer domain as a generic linear-feasibility problem. The existing `symbolic_domain.py` includes exact simplex calls, free-variable splitting, caching, fallback paths, and workarounds for invalid SymPy simplex results.

Replace that with a domain-specific exact optimizer.

For an affine expression,

[
a_0+\sum_i a_i c_i,
]

over an independent box, the minimum and maximum follow directly from coefficient signs.

For a group with a lower-total constraint,

[
\ell_i\le c_i\le u_i,
\qquad
\sum_i c_i\ge C_{\min},
]

the extremum is an exact fractional-knapsack calculation:

1. Start from the coefficient-preferred box corner.
2. Check whether its total satisfies (C_{\min}).
3. If not, add concentration in increasing order of objective cost.
4. Stop when the total minimum is reached.

This is exact, deterministic, and linearithmic in the number of species in the group.

Implement a memoized affine extractor:

```python
def affine_form(expr: Expr) -> AffineForm | None:
    ...
```

It should recursively support:

* symbols;
* constants;
* addition;
* multiplication of an affine expression by a constant.

Do not call `Poly` repeatedly on large expressions just to discover that they are not affine.

This one optimizer can provide:

* exact sign bounds for affine subexpressions;
* total-concentration margins;
* exact feasibility witnesses;
* parameter-box bounds;
* exact counterexamples for linear violations.

---

# 6. General expression analysis, without reaction-specific primitives

I agree that the checker should not require semantic wrappers such as `positive_floor()` or `bounded_fraction()`.

Reaction authors should be able to write any ordinary SymPy expression. Helper functions in reaction modules may still be used for readability, but they must return ordinary expressions and carry no privileged proof meaning.

The checker does, however, need generic mathematical knowledge about SymPy operations. That is not a kinetic primitive system. It is the minimum required to reason about expressions.

## 6.1 Reusable expression-analysis service

Create one analysis service owned by the run context:

```python
class ExpressionAnalyzer:
    def bounds(self, expression, domain) -> BoundResult: ...
    def sign(self, expression, domain) -> SignResult: ...
    def defined(self, expression, domain) -> DefinednessResult: ...
    def lipschitz(self, expression, domain) -> LipschitzResult: ...
```

All methods must memoize by:

```text
(expression, domain identity, active variables)
```

The same rate expression on the same domain should not be traversed independently by definedness, sign, and Lipschitz checks.

A useful internal result is:

```python
@dataclass(frozen=True)
class ExpressionFacts:
    interval: BoundResult
    sign: SignResult
    defined: ProofVerdict
    decisive_subexpression: Expr | None
```

Do not eagerly compute every property for every expression. Use lazy memoized methods.

## 6.2 Supported generic rules

The first implementation should explicitly support:

* `Symbol`;
* numeric constants;
* `Add`;
* `Mul`;
* `Pow`;
* `Abs`;
* `Min`;
* `Max`;
* `exp`;
* `log`;
* `sin`;
* `cos`;
* `sinh`;
* `cosh`;
* `tanh`;
* `atan`;
* ordinary rational expressions.

`Piecewise`, `floor`, `ceiling`, `sign`, discontinuous functions, and unfamiliar functions may return `UNKNOWN` unless a general derivative fallback certifies them.

No unsupported expression should cause an unbounded SymPy call.

## 6.3 Bounds

The bound engine should use:

1. exact affine bounds when possible;
2. interval composition otherwise;
3. small targeted algebraic rewrites only as fallback.

For products, use all endpoint products. For sums, add intervals. For monotone functions such as `exp` and `log`, transform endpoint bounds after proving the domain requirement.

Correlations may make interval bounds loose. Loose but sound bounds are acceptable. They produce `UNKNOWN`, not false failures.

## 6.4 Sign proofs

A sign proof should proceed in this order:

1. exact constants;
2. interval lower and upper bounds;
3. structural product and sum rules;
4. SymPy’s direct sign attributes under domain-derived assumptions;
5. `factor_terms()` only for a small unresolved expression;
6. exact witness search for disproof;
7. otherwise `UNKNOWN`.

Do not call `factor()` on every full rate expression. Do not call global `simplify()`.

## 6.5 Definedness

Definedness should be checked recursively.

Examples:

* reciprocal: denominator must be separated from zero;
* logarithm: argument must have a strictly positive lower bound;
* noninteger power: base must meet the appropriate real-domain condition;
* negative power: base must be separated from zero;
* trigonometric poles: relevant denominator must be separated from zero;
* `Abs`, `Min`, and `Max`: defined when their arguments are defined.

The output should identify the first unresolved or violated guard:

```text
FAIL
  denominator CH4 + CO + ... can equal zero
```

rather than printing an entire expanded rate.

## 6.6 Generic derivative fallback

For an otherwise unsupported but differentiable expression:

1. check an expression-size budget;
2. differentiate only with respect to active concentration symbols;
3. bound each derivative using the same bound engine;
4. return a Lipschitz certificate if every derivative has a finite absolute bound.

This gives broad coverage without defining every mathematical function manually.

Do not differentiate complete network source expressions. Differentiate rates individually and derive network constants through stoichiometry.

---

# 7. Lipschitz contract

This part needs unusually precise semantics.

## 7.1 What is being proved

For a scalar rate (r(c;T,P)), the physical-domain check should certify:

> There exists an open set (U\supset D) on which (r) is real-valued and Lipschitz in the concentration variables, uniformly for all configured (T) and (P).

Use the concentration-space infinity norm:

[
|c-c'|_\infty.
]

The returned constant (L_r) must satisfy:

[
|r(c;T,P)-r(c';T,P)|
\le
L_r|c-c'|_\infty
]

for all (c,c') in the declared closed domain and all allowed (T,P).

Any certified upper bound is a valid Lipschitz constant. It need not be the smallest possible constant. Name the field:

```python
constant_bound
```

and render it as “Certified Lipschitz constant” for users.

## 7.2 Open-neighbourhood requirement

A rate must not pass merely because it is one-sided well behaved on the physical domain.

Examples with physical lower bound (c_i=0):

| Expression                                | Result | Reason                                    |      |                                 |
| ----------------------------------------- | ------ | ----------------------------------------- | ---- | ------------------------------- |
| (c_i)                                     | PASS   | Globally Lipschitz                        |      |                                 |
| (c_i^2) on bounded domain                 | PASS   | Smooth on a neighbourhood                 |      |                                 |
| (                                         | c_i    | )                                         | PASS | Globally Lipschitz despite kink |
| (\max(c_i,0))                             | PASS   | Globally Lipschitz                        |      |                                 |
| (1/c_i)                                   | FAIL   | Undefined at the boundary                 |      |                                 |
| (\log c_i)                                | FAIL   | Undefined at the boundary                 |      |                                 |
| (\sqrt{c_i})                              | FAIL   | No real open neighbourhood around (c_i=0) |      |                                 |
| (c_i^{3/2})                               | FAIL   | No real open neighbourhood around zero    |      |                                 |
| (\sqrt{c_i^2+\epsilon}) with (\epsilon>0) | PASS   | Positive margin                           |      |                                 |

A reciprocal may pass only when its denominator has a strict margin from zero over the selected domain.

Therefore:

* (1/c_i) fails when (c_i) can equal zero;
* (1/(c_i+1)) may pass;
* (1/\sum_{i\in G}c_i) passes on a chamfered domain with (C_{G,\min}>0);
* the same expression fails on an independent physical box containing the all-zero gas state.

## 7.3 Proving an open neighbourhood exists

The first rewrite does not need to construct the maximal geometric neighbourhood radius.

It should prove neighbourhood existence by showing that every domain-sensitive guard has a strict margin on the closed domain. For example:

[
\inf_D |g(c)| = m > 0
]

for a denominator (g).

Because the domain is compact when its upper bounds are finite, and the supported primitives are continuous or globally Lipschitz on the relevant guarded range, a strict margin proves that some open neighbourhood exists.

Return:

```python
@dataclass(frozen=True)
class LipschitzCertificate:
    domain: DomainKind
    norm: str
    constant_bound: Expr
    active_variables: tuple[Symbol, ...]
    uniform_parameters: tuple[Symbol, ...]
    guard_margins: tuple[GuardMargin, ...]
```

An explicit neighbourhood radius can be added later. It is not necessary for the first correct implementation.

## 7.4 Compositional Lipschitz bounds

For the infinity norm, use these sound bounds.

Let:

[
B(f)=\sup_D |f|,
\qquad
L(f)=\text{certified Lipschitz constant}.
]

Then:

### Addition

[
L\left(\sum_i f_i\right)
\le
\sum_i L(f_i).
]

### Product

[
L\left(\prod_i f_i\right)
\le
\sum_i
L(f_i)\prod_{j\ne i}B(f_j).
]

### Reciprocal

If:

[
\inf_D |f|=m>0,
]

then:

[
L(1/f)\le \frac{L(f)}{m^2}.
]

### Exponential

[
L(e^f)
\le
e^{\sup_D f}L(f).
]

### Logarithm

If:

[
\inf_D f=m>0,
]

then:

[
L(\log f)\le \frac{L(f)}{m}.
]

### Absolute value

[
L(|f|)\le L(f).
]

### Minimum and maximum

[
L(\max(f_1,\ldots,f_n))
\le
\max_i L(f_i),
]

and likewise for `Min`.

### Rational power

For (f^\alpha), use the derivative bound on the proved positive interval of (f). A noninteger exponent requires a strictly positive base margin for the open-neighbourhood claim.

## 7.5 Network-source Lipschitz constant

Do not reanalyse the expanded source vector.

Given:

[
F(c)=Sr(c)
]

and per-rate constants (L_j), derive for each source component:

[
L_{F_i}
\le
\sum_j |\nu_{ij}|L_j.
]

Under the output infinity norm:

[
L_F
\le
\max_i\sum_j|\nu_{ij}|L_j.
]

This gives a network-vector-field constant essentially for free and directly reuses the rate-level Lipschitz results.

---

# 8. Shared objects and avoiding repeated work

The check DAG and the analysis cache should be separate concepts.

* The DAG expresses logical dependencies between user-visible checks.
* The run context owns reusable mathematical objects and lazy analysis providers.

Do not build a generic string-keyed dependency container. Use a typed context with cached properties.

```python
@dataclass
class AnalysisContext:
    case: Case

    @cached_property
    def physical_domain(self) -> ConcentrationDomain:
        ...

    @cached_property
    def augmented_domain(self) -> ConcentrationDomain:
        ...

    @cached_property
    def stoichiometry(self) -> Stoichiometry:
        ...

    @cached_property
    def network(self) -> ReactionNetwork:
        ...

    @cached_property
    def expression_analyzer(self) -> ExpressionAnalyzer:
        ...
```

Useful memoized methods:

```python
context.rate_facts(reaction, domain)
context.restricted_domain(domain, symbol, upper=0)
context.restricted_rate(reaction, substitutions)
context.source_contributions(species_id)
```

Objects that must be constructed once per run:

* physical domain;
* augmented domain;
* exact stoichiometric matrix;
* sparse source contributions;
* reaction-family shared terms;
* expression bounds for each `(expression, domain)`;
* rate Lipschitz facts;
* depletion substitutions;
* conserved-quantity basis.

The current context already attempts to cache some analysis by string keys, but checks still construct domains and expression facts independently.  The replacement should make shared work explicit and typed.

---

# 9. Check DAG

The current registry is only an ordered tuple and the runner executes every check regardless of earlier outcomes.

Replace it with an explicit static DAG.

```text
case loading and model validation
                │
                ▼
       ┌───────────────────┐
       │ chemistry gate    │
       │ atoms + mass      │
       └───────────────────┘
                │
       stop entire run on failure
                │
                ▼
      physical-domain definedness
          ┌─────┼───────────────┐
          ▼     ▼               ▼
 nonnegative  Lipschitz    zero at depletion
          │     │               │
          └─────┼───────────────┘
                ▼
      physical boundary inward
                │
                ▼
       forward invariance

                │
                ▼
      augmented definedness
                │
                ▼
      augmented Lipschitz
                │
                ▼
  negative-side non-repulsion
      + strict-attraction diagnostic

Nonblocking branches after chemistry gate:
  conserved quantities
  structural faces
  steady-state equations
```

## 9.1 Check metadata

Use a small explicit object:

```python
@dataclass(frozen=True)
class CheckSpec:
    id: str
    name: str
    stage: Stage
    scope: CheckScope
    requires: tuple[str, ...]
    blocking: bool
    profiles: frozenset[str]
    run: CheckRunner
```

Do not use decorators or automatic module discovery.

At startup, validate:

* unique IDs;
* every dependency exists;
* no cycles;
* deterministic topological ordering.

## 9.2 Result statuses

Use:

```text
PASS
FAIL
UNKNOWN
SKIPPED
ERROR
```

Interpretation:

* `PASS`: property proved;
* `FAIL`: property disproved by a proof or exact counterexample;
* `UNKNOWN`: proof engine was inconclusive;
* `SKIPPED`: prerequisite failed or required configuration was absent;
* `ERROR`: unexpected implementation failure.

An unexpected exception is not mathematical indeterminacy. The current runner catches all exceptions and turns them into `INDETERMINATE`; remove that behaviour.

Each check also has a role:

```text
BLOCKING
ADVISORY
ANALYSIS
```

Only blocking checks affect the overall result.

## 9.3 Stage-level stopping

Default behaviour should be `fail_fast: stage`.

For the chemistry gate:

1. run atom conservation on every reaction;
2. run mass conservation on every reaction;
3. report all failures in that stage;
4. terminate all later stages if any failed.

This avoids reporting only the first malformed reaction while still preventing expensive symbolic work on an invalid case.

Within later reaction-scoped checks, skip only the affected reaction where practical. For example, if one rate is undefined, other reactions can still receive sign and Lipschitz results. A case-level invariance certificate requires all relevant reaction-level prerequisites to pass.

## 9.4 Dependency reuse

A dependency result is never rerun.

The runner maintains:

```python
results: dict[str, CheckResult]
```

When a check is selected more than once through multiple profiles or dependency paths, return the existing result.

A downstream check should consume the structured dependency result rather than independently restating the same theorem.

---

# 10. Detailed checks

## 10.1 Case loading and model validation

This occurs before the DAG.

Validate:

* YAML shape and unknown keys;
* species existence;
* reaction selector syntax;
* requested reaction family import;
* reaction builder execution;
* duplicate IDs;
* symbol ownership;
* domain bounds;
* domain feasibility;
* gas and solid group membership;
* reserved names such as `temperature` and `pressure`;
* no broken selected family.

Only requested reaction families should be imported.

The current registry scans every module at import time. That exposes incomplete families such as the current `numaguchi.py`, even when they are not selected.

Replace automatic discovery with requested-family loading.

## 10.2 Atom conservation

For each element (e) and reaction (j):

[
\sum_i \nu_{ij}^{\mathrm{reactant}}a_{ie}
=========================================

\sum_i \nu_{ij}^{\mathrm{product}}a_{ie}.
]

All values are exact rationals. No tolerance is needed.

A fractional atom count such as (47/50) is treated exactly.

Output only imbalanced elements on failure.

## 10.3 Mass conservation

Use the declared molar masses in kg/mol and rational stoichiometry.

Because molar masses may be rounded physical data, retain a small configured relative and absolute tolerance. Do not introduce unit metadata or conversion.

Missing molar mass is a chemistry-gate failure, not an unavailable optional check.

## 10.4 Physical-domain rate definedness

For every reaction, prove the rate is real and finite everywhere on the physical domain.

This check uses `context.rate_facts(reaction, physical_domain)`.

It should report:

* first violated domain guard;
* exact violating point where available;
* concise subexpression;
* no full-rate dump by default.

## 10.5 Physical-domain nonnegativity

Require:

[
r_j(c;T,P)\ge0
]

throughout the physical domain.

Dependencies:

```text
physical_definedness
```

Proof order:

1. interval lower bound;
2. structural sign;
3. small symbolic fallback;
4. exact counterexample search;
5. `UNKNOWN`.

A directional rate proven negative anywhere in the physical domain is a failure.

## 10.6 Physical-domain Lipschitz

Use the contract in Section 7.

Return per reaction:

* pass/fail/unknown;
* certified constant bound;
* active concentration variables;
* decisive guard margins;
* optional numeric evaluation of the symbolic bound.

A rate such as (1/c_i) must fail when (c_i=0) belongs to the physical domain.

## 10.7 Zero at depletion

For each reaction and every consumed reactant or declared catalyst (s), check:

[
c_s=0 \implies r_j=0
]

on the physical domain.

Use exact substitution of the case-owned symbol:

```python
rate.subs({case.symbols.concentration(species_id): 0}, simultaneous=True)
```

Do not match by symbol name.

Do not call global `simplify()` or `equals()` by default. Use:

1. direct evaluated substitution;
2. structural zero rules;
3. numerator/denominator analysis;
4. small fallback.

If substitution creates (0/0) or an undefined expression, this is a failure, not a zero.

## 10.8 Physical boundary inward

Derive this result instead of symbolically expanding (Sr).

For concentration (c_i=0),

[
F_i
===

\sum_j\nu_{ij}r_j.
]

Every consuming reaction has (\nu_{ij}<0). If zero-at-depletion passed, those rates vanish at (c_i=0). Every remaining contribution has nonnegative stoichiometry and a nonnegative physical-domain rate. Therefore:

[
F_i\ge0.
]

This check should be a short theorem composition over previous structured results.

Dependencies:

```text
physical_rate_nonnegative
zero_at_depletion
```

It should not invoke SymPy.

## 10.9 Forward invariance

Combine:

* physical boundary inward;
* physical rate Lipschitz;
* derived source-vector Lipschitz constant.

The result is:

> The nonnegative concentration orthant is forward invariant for the reaction-source ODE throughout the declared physical domain, subject to the configured parameter ranges.

Do not use the stronger term “persistence” unless a separate persistence theorem is actually proved.

Dependencies:

```text
physical_boundary_inward
physical_lipschitz
```

## 10.10 Augmented-domain definedness

Repeat rate definedness on the augmented domain.

This is deliberately separate from physical-domain definedness. A rate may be entirely valid physically and undefined after a small negative concentration excursion.

## 10.11 Augmented-domain Lipschitz

Repeat the open-neighbourhood Lipschitz analysis on the augmented domain.

Examples:

* `sqrt(c)` with a negative excursion: fail;
* `1/c` where the interval crosses zero: fail;
* `1/sum(gas)` with a positive gas-total minimum: potentially pass;
* `Abs(c)`: pass;
* `Max(c, 0)`: pass.

This check is the regularity prerequisite for recovery analysis.

## 10.12 Negative-side non-repulsion

For every concentration (c_i) with an allowed negative excursion, prove:

[
c_i\le0
\quad\Longrightarrow\quad
F_i(c)\ge0
]

throughout the augmented domain.

Every other concentration retains its full augmented interval. Therefore simultaneous negative solver errors are included automatically.

Dependencies:

```text
augmented_definedness
augmented_lipschitz
```

However, apply dependencies per source component where possible. A species result only depends on reactions whose stoichiometric coefficient for that species is nonzero.

### Sparse source contributions

Do not immediately construct:

[
F_i=\sum_j\nu_{ij}r_j
]

as one expanded SymPy expression.

Store:

```python
source_contributions[i] = (
    (nu_i1, rate_1),
    (nu_i2, rate_2),
    ...
)
```

For the restricted domain (c_i\le0):

1. determine the sign and interval of each rate;
2. multiply by the exact stoichiometric coefficient;
3. sum lower bounds;
4. prove (F_i\ge0) if the total lower bound is nonnegative;
5. only construct the unresolved residue;
6. use a small symbolic fallback;
7. search for an exact counterexample;
8. otherwise return `UNKNOWN`.

This should detect:

[
A+B\rightarrow C,
\qquad
r=kAB
]

because when both (A) and (B) are negative, the source for (A) is outward.

### Exact counterexamples

Exact rational witness points may be used to disprove a universal property. That is not numerical sampling.

The domain should construct feasible exact points while respecting total constraints. Candidate generation should be bounded and directed by expression dependencies:

* canonical feasible point;
* lower bound;
* zero where allowed;
* upper bound;
* sign-relevant combinations for unresolved products.

A sampled point may prove failure but never prove success.

### Lower excursion boundary

A separate lower-face check is unnecessary.

If:

[
c_i\le0\implies F_i\ge0
]

is proved throughout the complete augmented interval, it already includes:

* (c_i=0);
* (c_i=\ell_i^{\mathrm{exc}});
* every point in between.

The property simultaneously proves physical-boundary non-repulsion under augmented errors and prevents the reaction source from pushing the concentration below the checked excursion floor.

### Strict attraction

Also attempt the stronger diagnostic:

[
c_i<0
\quad\Longrightarrow\quad
F_i(c)>0.
]

This does not affect the main pass result.

Report:

```text
non-repelling: proved
strictly attracting: proved / disproved / unknown
```

Do not claim finite-time re-entry. A Lipschitz source that vanishes at zero will often allow only asymptotic return.

---

# 11. Nonblocking analyses

These must not change whether the reaction implementation is physically valid.

## 11.1 Conserved quantities

Keep:

* exact sparse stoichiometric matrix;
* rank;
* connected components;
* unchanged species;
* a compact exact basis of the left nullspace.

Do not compute all nonnegative extreme rays by default.

The current conservation module includes a substantial exact double-description implementation. That can remain as an explicitly selected advanced analysis or be removed until needed.

Default output should show a simple exact basis.

## 11.2 Structural invariant and dead faces

Replace expression-heavy face enumeration with reaction-network structure where possible.

For a depleted species set (D):

* a structural dead face is one where every reaction has at least one required reactant or catalyst in (D);
* a structural invariant face is one where every reaction capable of producing a depleted species is disabled by at least one required species in (D).

This is a hypergraph problem based on reaction supports. It does not depend on the algebraic complexity of rates.

These are sufficient structural certificates. Additional faces arising from algebraic cancellation are not required in the initial rewrite.

Use an output limit because the number of minimal hitting sets may still be exponential.

## 11.3 Steady-state equations

Remove the large branch compiler and custom equilibrium formatter from the default design.

The initial replacement should:

1. build sparse (F=Sr);
2. select a low-complexity independent set of stoichiometric rows;
3. report the corresponding equations (F_i=0);
4. retain expressions structurally;
5. optionally identify trivial shared zero factors;
6. expose full expressions through JSON.

Do not attempt general explicit solving or complete branch decomposition.

The current equilibrium module mixes mathematical compilation, branching, CSE, aliases, line wrapping, and text rendering in one large file.

That is outside the core responsibility of a physical reaction-expression checker.

---

# 12. Reaction-family loading

Each family remains one Python file.

Use one family-level builder so common expressions are constructed once:

```python
def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    terms = xu_froment_terms(symbols)

    return {
        "smr_fw": Reaction(...),
        "smr_bw": Reaction(...),
        "wgs_fw": Reaction(...),
        ...
    }
```

A case selector may name:

```text
xu_froment
xu_froment.smr_fw
```

Loading process:

1. resolve only the requested family module;
2. call `build_family()` once;
3. validate its complete returned mapping;
4. select requested reactions;
5. reject overlapping selections.

Do not scan every module in a package.

Support both:

```text
built-in reaction families
case_directory/reactions/<family>.py
```

Document that local reaction modules execute trusted Python code.

---

# 13. Proposed case format

```yaml
schema: 1

species:
  - Ar
  - CH4
  - CO
  - CO2
  - H2
  - H2O
  - O2
  - Ni
  - NiO
  - CaAl2O4

inerts:
  - Ar
  - CaAl2O4

reactions:
  - medrano
  - xu_froment

parameters:
  temperature: [573.15, 1473.15]
  pressure: [100000.0, 3000000.0]

domain:
  concentration_model: chamfered

  upper:
    default: 1000.0
    overrides:
      Ni: 10000.0
      NiO: 10000.0
      CaAl2O4: 10000.0

  excursion_lower:
    default: -0.05
    overrides:
      CaAl2O4: -0.001

  totals:
    gas:
      mode: ideal_gas_minimum

    solid:
      mode: explicit
      species:
        - Ni
        - NiO
      value: 1.0e-8

checks:
  profile: robust
  include:
    - conserved_quantities
    - structural_faces
  exclude:
    - steady_state_equations
  fail_fast: stage

report:
  verbosity: failures
  format: text
```

For independent concentrations:

```yaml
domain:
  concentration_model: independent

  upper:
    default: 1000.0

  excursion_lower:
    default: -0.05
```

Unknown keys should be rejected rather than silently ignored.

## 13.1 Profiles

Suggested profiles:

| Profile    | Included work                                                                           |
| ---------- | --------------------------------------------------------------------------------------- |
| `basic`    | Loading, model validation, atom conservation, mass conservation                         |
| `physical` | `basic` plus physical definedness, sign, Lipschitz, depletion, invariance               |
| `robust`   | `physical` plus augmented definedness, augmented Lipschitz, negative-side non-repulsion |
| `analysis` | `physical` plus conserved quantities, structural faces, steady-state equations          |
| `all`      | `robust` plus all analyses                                                              |

Default profile: `physical`.

The example reforming case should explicitly select `robust`.

Selection rules:

* `include` adds a check and its transitive prerequisites;
* `exclude` removes a check;
* explicitly excluding a prerequisite of an explicitly included check is a configuration error;
* duplicate selection does not rerun a check.

## 13.2 CLI

Keep the CLI modest:

```text
rxn-checker CASE
rxn-checker CASE --profile robust
rxn-checker CASE --checks physical_lipschitz,augmented_lipschitz
rxn-checker CASE --skip steady_state_equations
rxn-checker CASE --format json
rxn-checker CASE --output report.json
rxn-checker --list-checks
```

Do not write a report file unless `--output` or a configured output path is supplied.

Useful exit codes:

```text
0  all selected blocking checks passed
1  at least one blocking check failed or was unknown
2  invalid case or internal error
```

A `--debug` option may re-raise internal exceptions.

---

# 14. Structured results and reporting

## 14.1 Core result types

```python
class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class Role(StrEnum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"
    ANALYSIS = "analysis"
```

```python
@dataclass(frozen=True)
class Finding:
    subject: str
    verdict: Verdict
    summary: str
    evidence: Evidence | None = None
```

```python
@dataclass(frozen=True)
class CheckResult:
    check_id: str
    role: Role
    findings: tuple[Finding, ...]
    duration_seconds: float
```

```python
@dataclass(frozen=True)
class RunResult:
    case_name: str
    selected_checks: tuple[str, ...]
    results: Mapping[str, CheckResult]
    overall: Verdict
```

Do not create a different public result dataclass for every simple check unless it carries genuinely useful structured mathematical data.

Specialized internal evidence objects are appropriate for:

* exact imbalance;
* interval bound;
* exact counterexample;
* Lipschitz certificate;
* conserved quantity.

## 14.2 Human output

Default output should collapse passing reaction-wide checks:

```text
rxn-checker: FAIL
Case: reforming_case
Profile: robust

Chemistry
  PASS  Atom conservation                  9/9 reactions
  PASS  Mass conservation                  9/9 reactions

Physical domain
  PASS  Rate definedness                   9/9 reactions
  PASS  Rate non-negativity                9/9 reactions
  FAIL  Lipschitz continuity               8/9 reactions
        xu_froment.smr_fw
        denominator H2 reaches zero on the physical boundary

  SKIP  Forward invariance
        requires physical_lipschitz

Augmented domain
  SKIP  Augmented Lipschitz
        physical stage did not pass
  SKIP  Negative-side non-repulsion
        requires augmented_lipschitz

Overall: FAIL
```

On a successful Lipschitz check:

```text
PASS  medrano.reduction_h2
      L∞ <= 3.42e4
      uniform for T in [...] and P in [...]
```

Verbose mode may include symbolic constants, guard margins, and exact expressions.

## 14.3 JSON output

JSON should preserve:

* selected domain;
* expression or expression identifier;
* exact symbolic bound as a string;
* numerical evaluation;
* guard margins;
* exact rational counterexample values;
* prerequisite status;
* timing.

The human renderer must not be the source of truth.

---

# 15. Suggested package structure

```text
src/rxn_checker/
  __init__.py
  model.py
  loading.py
  domain.py
  network.py

  proof/
    __init__.py
    bounds.py
    signs.py
    regularity.py
    affine.py
    evidence.py

  checks/
    __init__.py
    definitions.py
    chemistry.py
    physical_rates.py
    invariance.py
    robustness.py

  analyses/
    __init__.py
    conservation.py
    faces.py
    steady_states.py

  context.py
  results.py
  runner.py
  reporting.py
  cli.py

  reactions/
    __init__.py
    medrano.py
    xu_froment.py

  species/
    __init__.py
    registry.py
```

This is a guide, not a mandate to create many tiny files. Combine files where doing so improves linear readability.

The main separation to preserve is:

```text
model/domain
proof machinery
checks
analyses
execution
rendering
```

---

# 16. Readability rules for the rewrite

The implementing agent should actively optimize for a reader following the code top to bottom.

Use these rules:

1. No import-time scanning or registration magic.
2. No giant root `__init__.py` re-export list.
3. No renderer logic inside mathematical checks.
4. No generic string-keyed object caches.
5. No defensive `MappingProxyType` wrapping throughout the code.
6. Use frozen dataclasses where immutability matters; use ordinary local dictionaries internally.
7. A check runner should usually read as:

   * obtain prerequisite data;
   * perform the theorem-specific calculation;
   * construct findings;
   * return.
8. Put general symbolic rules in one place.
9. Put domain logic in one place.
10. Avoid repeated domain construction inside checks.
11. Avoid functions that both compute a result and format several pages of prose.
12. Avoid global `simplify`, `solve`, `equals`, and unrestricted factorisation.
13. Comment mathematical reasons, not obvious Python operations.
14. Prefer a few explicit branches over generic abstractions whose behaviour is difficult to trace.
15. Keep the stable public API small.

A reasonable target is that no ordinary check module exceeds roughly 300–400 lines. Algorithmic proof modules may be longer if the flow remains direct.

---

# 17. Performance strategy

Performance should come from selecting the right mathematics, not adding caches around expensive inappropriate algorithms.

## 17.1 Required optimizations

* build each selected family once;
* preserve shared SymPy subexpression objects;
* construct sparse stoichiometry once;
* store source contributions rather than expanded source sums;
* build each domain once;
* exact specialized affine optimization;
* memoize expression facts by expression and domain;
* compute per-rate Lipschitz constants once;
* derive source Lipschitz constants algebraically;
* use upstream check results directly;
* terminate later stages after chemistry-gate failure;
* bound all fallback work.

## 17.2 Fallback budgets

Operation budgets should apply only to expensive fallback methods, not to the entire check.

Examples:

```text
maximum nodes for derivative fallback
maximum factors for local factorisation
maximum exact witness candidates
maximum structural face results
```

When a budget is exceeded:

```text
UNKNOWN
```

with a clear diagnostic.

The tool must never appear to hang indefinitely because SymPy was asked an unrestricted global question.

## 17.3 Soft runtime targets

On the current reforming case:

```text
loading + chemistry gate          below 0.5 s
physical profile                  below 3–5 s
robust profile                    below 10 s
example case                      effectively immediate
```

These are engineering targets rather than formal correctness criteria. The key hard requirement is deterministic termination with bounded fallback work.

The current repository reports approximately 26 seconds for enabled reforming checks without the deleted recovery analysis.  The rewrite should materially improve on that.

---

# 18. Minimal local tests

No GitHub Actions, deployment pipeline, coverage gate, or large testing framework is needed.

A small local suite is still necessary because these functions make mathematical claims and subtle refactors can silently change them. The latest repository state deleted the former test suite entirely, which allowed stale imports of the deleted recovery module to remain in the public package.

Use `pytest` for concision, with approximately four files:

```text
tests/test_model_and_loading.py
tests/test_domains.py
tests/test_expression_analysis.py
tests/test_checks.py
```

Add one integration test for each example case.

## 18.1 Required mathematical examples

### Exact chemistry

* `1/2 O2` stoichiometry;
* Fe₀.₉₄O composition;
* exact atom balance pass;
* atom imbalance fail;
* mass imbalance fail;
* chemistry-gate failure skips later checks.

### Domains

* independent physical domain;
* independent augmented domain;
* gas chamfer;
* solid chamfer with explicit species list;
* ideal-gas-derived gas minimum;
* infeasible total minimum rejected;
* physical and augmented domains share upper and total constraints;
* augmented individual concentration may be negative while total remains positive.

### Lipschitz

* (r=c): pass, (L=1);
* (r=2c): pass, (L=2);
* (r=c^2) on ([0,u]): pass with finite bound;
* (r=|c|): pass;
* (r=\max(c,0)): pass;
* (r=1/c) with (c_{\min}=0): fail;
* (r=1/c) with (c_{\min}>0): pass;
* (r=1/\sum c_g) on independent physical box: fail;
* the same rate on chamfered gas domain: pass;
* (r=\sqrt c) touching zero: fail;
* (r=\sqrt{c^2+\epsilon}): pass;
* (r=\log c) touching zero: fail;
* unsupported discontinuous function: unknown.

### Invariance

* (A\to B,\ r=kA): physical invariance pass;
* consuming rate not zero at (A=0): fail;
* negative physical rate: fail.

### Augmented recovery

* (A\to B,\ r=kA): non-repelling and strictly attracting for negative (A);
* (A\to B,\ r=k\max(A,0)): non-repelling but not strictly attracting;
* (A\to B,\ r=k|A|): fail with worsening negative (A);
* (A\to B,\ r=k\sqrt A): recovery skipped because augmented regularity fails;
* (A+B\to C,\ r=kAB): simultaneous negative counterexample detected;
* inert negative species: non-repelling but stuck.

### DAG and caching

* shared network constructed once;
* physical rate facts computed once per rate/domain;
* selecting downstream check includes prerequisites;
* failed chemistry gate stops later stages;
* explicit dependency conflict is rejected;
* unexpected exception becomes `ERROR`.

This is enough. Do not test every private helper or target an arbitrary coverage percentage.

Local verification command:

```shell
uv run pytest -q
```

---

# 19. Rewrite phases

## Phase 0 — Make the current repository coherent

Before introducing the new architecture:

1. remove stale imports of deleted `nonphysical_recovery`;
2. remove stale exports from both package `__init__.py` files;
3. remove checked-in generated reports;
4. stop automatic report writing;
5. quarantine or delete incomplete `numaguchi.py`;
6. add one import smoke test;
7. ensure `rxn-checker --help` works.

The current package root and checks package still reference the deleted recovery implementation.

Do not spend time polishing the old architecture beyond making it usable during the rewrite.

## Phase 1 — New exact model and loader

Implement:

* rational parser;
* exact species composition;
* exact reaction stoichiometry;
* new `CaseSymbols`;
* bounded T/P parameters;
* requested-family loading;
* family-level build;
* model validation;
* new YAML schema.

Adapt the example reactions and reforming reactions to the new API.

Acceptance:

* both cases load;
* fractional stoichiometry and atom counts work;
* no unselected family is imported;
* family shared terms are built once.

## Phase 2 — Unified domains

Implement:

* `DomainSpec`;
* physical and augmented generation;
* independent geometry;
* chamfered geometry;
* explicit gas and solid minima;
* ideal-gas-derived gas minimum;
* exact feasibility;
* exact affine bounds;
* exact witness construction;
* domain restriction.

Acceptance:

* all domain tests pass;
* every check obtains domains only from `AnalysisContext`;
* no check constructs its own interpretation of the domain.

## Phase 3 — Results, context, DAG, and selection

Implement:

* result enums and dataclasses;
* typed `AnalysisContext`;
* static check registry;
* dependency expansion;
* profiles;
* stage stopping;
* check selection from YAML and CLI;
* text and JSON skeleton renderers.

Initially, use placeholder or adapted checks where necessary.

Acceptance:

* dependency planning is deterministic;
* duplicate paths do not rerun checks;
* failed prerequisites produce `SKIPPED`;
* internal errors are distinct from `UNKNOWN`.

## Phase 4 — Generic bound and sign engine

Implement:

* affine extraction;
* exact affine extrema;
* interval bounds;
* generic arithmetic rules;
* generic function rules;
* sign proofs;
* definedness guards;
* concise evidence;
* bounded symbolic fallback;
* exact failure witnesses.

Acceptance:

* no unrestricted `simplify`, `factor`, `solve`, or `equals`;
* required expression-analysis tests pass;
* repeated subexpressions are cached.

## Phase 5 — Lipschitz engine

Implement:

* open-neighbourhood guard logic;
* compositional constants;
* rational-power handling;
* parameter-uniform bounds;
* derivative fallback;
* per-rate certificates;
* derived network source constant.

Acceptance:

* all examples in Section 18 pass;
* `1/c` fails at a zero boundary;
* `Abs` and `Max` pass;
* constants are reported;
* physical and augmented checks use the explicitly selected domain.

## Phase 6 — Core physical checks

Rewrite:

* atom conservation;
* mass conservation;
* physical definedness;
* rate nonnegativity;
* zero at depletion;
* boundary inward theorem;
* forward-invariance certificate.

Acceptance:

* chemistry failures stop the run;
* physical invariance is derived without expanding source expressions;
* report remains concise on the reforming case.

## Phase 7 — Augmented robustness

Implement:

* augmented definedness;
* augmented Lipschitz;
* sparse negative-side non-repulsion;
* strict-attraction diagnostic;
* exact counterexample search;
* inert-stuck reporting.

Acceptance:

* simultaneous negative errors are included;
* no negative-species powerset enumeration;
* no finite-time return claim;
* reforming robust profile terminates predictably.

## Phase 8 — Analyses

Implement or simplify:

* compact conserved-quantity basis;
* structural faces;
* independent steady-state equations.

Do this only after the core physical checker is stable.

## Phase 9 — Delete legacy code

Remove:

* old generic symbolic-domain implementation;
* old Lipschitz module;
* old nonnegative-rate module;
* old negative-side recovery module;
* old zero-at-depletion module;
* old runner and result framework;
* large equilibrium branch compiler;
* symbolic terminal-face search;
* stale recovery design references from the README;
* generated reports;
* unused exports.

The advanced deleted recovery design may be retained as a clearly marked design note under:

```text
docs/legacy/nonphysical_recovery_regions.md
```

It must not be documented as implemented functionality.

## Phase 10 — Documentation and benchmark

Produce:

* concise README;
* case-schema documentation;
* check dependency diagram;
* exact statement of each proof;
* explanation of physical versus augmented domains;
* Lipschitz norm and constant semantics;
* SI-only notice;
* benchmark of example and reforming cases;
* brief limitations section.

Do not write a long internal implementation tutorial into the main README.

---

# 20. Deliberate exclusions

The agent should not add any of the following during this rewrite:

* unit libraries or dimensional analysis;
* reversible-reaction objects;
* reaction-pair detection;
* numerical sampling as a proof method;
* interval subdivision or Monte Carlo checking;
* numerical integration;
* Jacobian eigenvalue or stiffness analysis;
* rate-comparison metrics;
* numerical performance comparisons;
* arbitrary plugin systems;
* dynamic check discovery;
* arbitrary domain constraints;
* arbitrary overlapping total-concentration groups;
* exact equilibrium solving;
* GitHub Actions;
* deployment configuration;
* coverage gates;
* compatibility layers for every current internal API.

The architecture should leave room for later numerical checks, but no framework should be built solely for hypothetical future features.

---

# 21. Definition of done

The rewrite is complete when all of the following hold.

## Model and configuration

* exact rational reaction coefficients;
* fractional atom counts supported;
* T/P represented as bounded parameters;
* independent and chamfered concentration models supported;
* physical and augmented domains generated from one specification;
* explicit gas and solid total minima supported;
* ideal-gas minimum supported;
* user-selectable checks and profiles supported.

## Check flow

* static DAG;
* shared objects built once;
* chemistry failures terminate later stages;
* downstream checks consume upstream results;
* no duplicate execution;
* skipped checks explain the failed prerequisite.

## Physical mathematics

* atom and mass conservation;
* physical rate definedness;
* physical rate nonnegativity;
* physical open-neighbourhood Lipschitz check;
* per-rate Lipschitz constants;
* source-vector Lipschitz constant;
* zero at depletion;
* boundary inward certificate;
* forward-invariance certificate;
* augmented definedness;
* augmented open-neighbourhood Lipschitz check;
* negative-side non-repulsion;
* strict-attraction diagnostic.

## Performance and robustness

* no unrestricted global SymPy operation;
* no expanded source expression unless required as a bounded fallback;
* deterministic termination;
* reforming case completes in practical time;
* concise default report.

## Code quality

* substantially less production code than the current collection of large symbolic modules;
* no stale imports;
* no incomplete exposed reaction family;
* no generated report committed;
* minimal local theorem tests pass;
* no CI machinery;
* README accurately describes only implemented features.

---

# 22. Final directive to the implementing agent

The rewrite should prioritize **proof composition over general symbolic search**.

Use:

* exact chemistry;
* simple domain geometry;
* compositional expression facts;
* sparse stoichiometric contributions;
* explicit dependencies;
* cached shared analyses.

Avoid trying to make SymPy independently rediscover the complete mathematical meaning of every expanded rate and source expression.

Where a property cannot be proved efficiently:

```text
return UNKNOWN with the decisive unresolved subexpression
```

rather than expanding, simplifying, solving, or enumerating until the process becomes unusable.

The target is not maximal theorem-proving power. It is a reaction-expression checker whose conclusions are mathematically sound, whose limitations are visible, and whose code can be understood by reading it in a reasonably straight line.
