# Implementation Plan: Differential Solver Profile

## 1. Purpose

Add a nonblocking `differential_solver_profile` analysis to `rxn-checker` that characterises how reaction kinetics are likely to affect:

- IDAS time-step pressure;
- Newton convergence;
- reuse of an out-of-date Jacobian;
- kinetic coupling between species and reactions;
- formation of fast stable or amplifying modes;
- robustness near physical boundaries and during small nonphysical solver excursions.

The analysis remains **fully symbolic and SymPy-only**. It does not:

- import DAETools;
- run a simulation;
- sample numerical states;
- construct the complete packed-bed DAE;
- claim an actual IDAS step size or stiffness ratio.

Its object of study is the local kinetic system

\[
\dot{\mathbf c}=F(\mathbf c,T,P)=S\,r(\mathbf c,T,P),
\]

together with the derivatives of \(r\) and \(F\) over the configured physical and augmented domains.

The profile should answer:

1. Is each rate smooth, piecewise smooth, or merely Lipschitz?
2. Where are its kinks and derivative singularities?
3. Which concentrations, temperature, and pressure affect it most strongly?
4. Is a reaction self-damping or self-amplifying as it advances?
5. Which reactions strongly affect one another?
6. Which reaction contributes most to fast kinetic modes?
7. How large can the source Jacobian become?
8. How rapidly can that Jacobian change?
9. What part of the result is certified, conservative, or unresolved?

---

## 2. Connection to the actual DAETools system

### 2.1 DAETools residual structure

`multisolid-CL` gives each kinetic rate its own algebraic variable \(R_j\). Species balances use those variables through the stoichiometric matrix, while separate equations impose

\[
R_j-r_j(\mathbf c,T,P)=0.
\]

The implemented equations are structurally:

\[
\begin{aligned}
F_c &= \dot{\mathbf c}-S\mathbf R+\text{transport}=0,\\
F_R &= \mathbf R-r(\mathbf c,T,P)=0.
\end{aligned}
\]

This is exactly how the current model creates species and solid balances and then creates one distributed `reaction_rate_*` equation for each rate. 

IDAS solves DAEs of the form

\[
F(t,y,\dot y)=0
\]

using variable-order BDF methods and Newton-type nonlinear iterations. Its iteration matrix is

\[
J_{\mathrm{IDA}}
=
\frac{\partial F}{\partial y}
+
\alpha\frac{\partial F}{\partial\dot y},
\]

where \(\alpha\) changes with the current step size and BDF coefficients.

Ignoring transport and the additional thermal and pressure variables for a moment, the local kinetic block is

\[
J_{\mathrm{kin}}(\alpha)
=
\begin{bmatrix}
\alpha I & -S\\
-R_c & I
\end{bmatrix},
\qquad
R_c=\frac{\partial r}{\partial\mathbf c}.
\]

Eliminating the algebraic rate correction gives the Schur complement

\[
\boxed{
M_{\mathrm{kin}}(\alpha)
=
\alpha I-SR_c
=
\alpha I-J_c
}
\]

with

\[
J_c=\frac{\partial F}{\partial\mathbf c}
=S\frac{\partial r}{\partial\mathbf c}.
\]

This is the main mathematical justification for the profile: the kinetic source Jacobian \(J_c\) appears directly in the Newton system after the rate variables are eliminated.

### 2.2 What the profile does not reconstruct

The complete DAETools Jacobian also includes:

- spatial transport;
- gas mole-fraction closures;
- pressure and ideal-gas closure;
- temperature and enthalpy equations;
- gas properties;
- boundary equations;
- coupling between neighbouring cells.

The current DAETools rate mirrors also use gas mole fractions directly, whereas `rxn-checker` uses canonical species concentration symbols and often constructs fractions from them.  

Therefore, the result must be labelled:

```text
kinetic concentration-space differential profile
```

not:

```text
complete DAETools Jacobian profile
```

Regularity, source-vector feedback, kinetic timescales, and stoichiometric coupling remain physically meaningful. Exact DAETools incidence, scaling, and full-system conditioning remain runtime-model properties.

---

## 3. Analyse the source-equivalent solver representation

The differential profile should reuse the source-equivalent flux construction from `evaluation_profile`.

`rxn-checker` currently represents Xu–Froment forward and backward directions as separate nonnegative rates, while `multisolid-CL` uses one signed net rate for each reversible reaction.  

If proportional stoichiometric columns are grouped as

\[
S=\begin{bmatrix}\nu_1&\cdots&\nu_m\end{bmatrix},
\qquad
\nu_j=\alpha_j\bar\nu_g,
\]

their combined source is represented by

\[
\bar\nu_g\,\bar r_g,
\qquad
\bar r_g=\sum_{j\in g}\alpha_jr_j.
\]

For a reversible pair,

\[
\bar r=r_{\mathrm{fw}}-r_{\mathrm{bw}}.
\]

The source vector and source Jacobian are unchanged:

\[
S_{\mathrm{declared}}r_{\mathrm{declared}}
=
S_{\mathrm{flux}}\bar r,
\]

\[
S_{\mathrm{declared}}
\frac{\partial r_{\mathrm{declared}}}{\partial\mathbf c}
=
S_{\mathrm{flux}}
\frac{\partial\bar r}{\partial\mathbf c}.
\]

Use:

- **declared reactions** for per-direction diagnostics;
- **source-equivalent fluxes** for reaction interaction, self-feedback, and solver-facing network analysis.

The helper that builds source-equivalent fluxes should live in `network.py` or another shared structural module. The differential profile should not depend on the `evaluation_profile` check result; both analyses should call the same pure helper.

---

## 4. Analyse both configured domains

Produce two domain profiles.

### 4.1 Physical domain

The physical profile describes the intended simulation states.

It should be used for:

- primary derivative bounds;
- source-Jacobian bounds;
- self-feedback;
- kinetic mode analysis;
- curvature and Jacobian-variation bounds.

### 4.2 Augmented domain

The augmented profile describes the configured negative-side excursion region.

It should emphasise:

- whether physical-domain branch simplifications cease to hold;
- new kinks or singular derivative surfaces;
- whether gradients become unbounded or unresolved;
- how much larger the derivative envelopes become;
- whether a rate loses \(C^1\) regularity immediately outside the physical state space.

The augmented profile is particularly relevant to Newton iterates and integration errors, but it must not claim that IDAS will necessarily visit every point in that domain.

### 4.3 No check prerequisites

Register the profile without dependencies on the definedness or Lipschitz checks.

The profile should call the shared `ExpressionAnalyzer` itself and return partial information when:

- one rate is undefined;
- a derivative cannot be bounded;
- the augmented domain is inconclusive;
- the Hessian budget is exhausted.

This keeps the analysis useful even when another check fails.

---

## 5. Profile layers

The complete profile should contain six layers.

---

## 5.1 Domain-aware regularity audit

### Branch reduction

Before differentiating, reduce branches that are provably redundant on the current domain.

Examples:

\[
\max(c_i,0)=c_i
\quad\text{when}\quad c_i\geq0,
\]

\[
|g|=g
\quad\text{when}\quad g\geq0,
\]

\[
\min(1,f)=f
\quad\text{when}\quad 0\leq f\leq1.
\]

Implement one recursive function:

```python
def reduce_branches(
    expression: sp.Expr,
    domain: ConcentrationDomain,
    analyzer: ExpressionAnalyzer,
) -> BranchReduction:
    ...
```

It should only reduce:

- `Abs`;
- `Min`;
- `Max`;
- simple `Piecewise` branches whose condition is provable.

Do not call broad `simplify()`, `refine()`, `piecewise_fold()`, or aggressive algebraic transformations.

Record every reduction:

```text
original subexpression
selected branch
proof basis
domain
```

This is important because many Medrano clamps may be redundant on the physical domain but active on the augmented domain.

### Surface extraction

Extract potential nonsmooth or derivative-singular surfaces generically.

| Expression | Surface |
|---|---|
| `Abs(g)` | \(g=0\) |
| `Max(a,b)` | \(a-b=0\) |
| `Min(a,b)` | \(a-b=0\) |
| `Piecewise` | boundaries of supported relational conditions |
| \(g^{-n}\) | \(g=0\) |
| \(\log g\) | \(g=0\) |
| \(g^p,\ p\notin\mathbb Z\) | \(g=0\) |
| \(\sqrt g\) | \(g=0\), where the first derivative may diverge |
| \(g^p,\ 1<p<2\) | \(g=0\), where the second derivative may diverge |

Use a small classification:

```python
class SurfaceLocation(StrEnum):
    EXCLUDED = "excluded"
    BOUNDARY = "boundary"
    INTERIOR = "interior"
    EVERYWHERE = "everywhere"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"
```

For an exact bound \([\ell,u]\):

```text
u < 0 or l > 0        → EXCLUDED
l < 0 < u             → INTERIOR
l = u = 0             → EVERYWHERE
l = 0 < u or l < 0 = u → BOUNDARY
```

For non-exact enclosures:

- a sign-definite enclosure may still prove `EXCLUDED`;
- an enclosure containing zero should normally produce `POSSIBLE`;
- exact witnesses on opposite sides may prove `INTERIOR`.

### Regularity classification

Per rate and domain, report:

```python
class Regularity(StrEnum):
    C11 = "C1,1"
    C1 = "C1"
    C1_INTERIOR = "C1 on physical interior"
    PIECEWISE_C1 = "piecewise C1"
    LIPSCHITZ = "Lipschitz only"
    CONTINUOUS = "continuous only"
    UNKNOWN = "unknown"
```

Interpretation:

- `C1,1`: first derivatives exist and have a certified Lipschitz bound;
- `C1`: continuously differentiable on the analysed domain;
- `C1_INTERIOR`: smooth away from the physical boundary;
- `PIECEWISE_C1`: smooth on branches but has reachable switch surfaces;
- `LIPSCHITZ`: suitable for uniqueness arguments but not classically differentiable everywhere;
- `CONTINUOUS`: continuity established, but no finite first-derivative envelope;
- `UNKNOWN`: available symbolic analysis is inconclusive.

Do not treat nonsmoothness as a failing chemistry check. It is a solver-facing property.

---

## 5.2 Sparse first-derivative envelopes

For source-equivalent rate \(r_j\), calculate only derivatives with respect to symbols actually present in the expression:

\[
G_{jk}
=
\sup_{x\in D}
\left|
\frac{\partial r_j}{\partial c_k}
\right|.
\]

Also calculate separate operating-variable sensitivities:

\[
G_{jT}
=
\sup_D
\left|
\frac{\partial r_j}{\partial T}
\right|,
\qquad
G_{jP}
=
\sup_D
\left|
\frac{\partial r_j}{\partial P}
\right|.
\]

### Generalise the existing Lipschitz engine

The current Lipschitz code recursively propagates one scalar derivative bound. Refactor its internal result from:

```python
@dataclass(frozen=True)
class _LipBound:
    constant: sp.Expr
```

to something equivalent to:

```python
@dataclass(frozen=True)
class GradientEnvelope:
    components: tuple[tuple[sp.Symbol, sp.Expr], ...]
    guards: tuple[GuardMargin, ...]

    @property
    def linfinity_lipschitz(self) -> sp.Expr:
        return sum(bound for _, bound in self.components)
```

The existing Lipschitz certificate then becomes a reduction of the same per-variable envelope.

This avoids maintaining two separate implementations of:

- product rules;
- power rules;
- `Abs`, `Min`, and `Max`;
- exponential and logarithm derivatives;
- guard margins;
- unsupported-function fallback.

### Absolute and signed derivative information

Maintain two levels:

1. **Absolute derivative envelope**

   Available for Lipschitz piecewise-smooth expressions, including `Abs`, `Min`, and `Max`.

2. **Signed derivative interval**

   Attempted after branch reduction when the expression is classically differentiable on the analysed domain:

   \[
   \frac{\partial r_j}{\partial c_k}
   \in
   [\underline g_{jk},\overline g_{jk}].
   \]

Use:

```python
derivative = sp.diff(reduced_expression, variable)
bounds = analyzer.bounds(derivative, domain)
```

Do not simplify the derivative globally.

If a signed interval is unavailable, retain the absolute envelope. That still supports conservative source-Jacobian magnitude bounds.

### Derivative cache

Use one top-level cached helper:

```python
@cache
def symbolic_derivative(
    expression: sp.Expr,
    variable: sp.Symbol,
) -> sp.Expr:
    return sp.diff(expression, variable)
```

Do not put nested differentiation functions inside the main profile routine.

---

## 5.3 Reaction-direction self-feedback

For source-equivalent flux \(j\), with stoichiometric direction \(\nu_j\), calculate

\[
\boxed{
\gamma_j
=
\nabla_{\mathbf c}r_j^\mathsf T\nu_j
}
\]

or explicitly,

\[
\gamma_j
=
\sum_i
\nu_{ij}
\frac{\partial r_j}{\partial c_i}.
\]

This is the derivative of the rate as its own reaction advances:

\[
\gamma_j
=
\left.
\frac{\mathrm d}{\mathrm d\xi}
r_j(\mathbf c+\nu_j\xi)
\right|_{\xi=0}.
\]

Classify its bound:

```python
class FeedbackKind(StrEnum):
    DAMPING = "uniformly damping"
    AMPLIFYING = "uniformly amplifying"
    MIXED = "changes sign"
    ZERO = "zero"
    UNKNOWN = "unknown"
```

Using an interval \([\underline\gamma_j,\overline\gamma_j]\):

```text
upper < 0        → DAMPING
lower > 0        → AMPLIFYING
lower < 0 < upper → MIXED
lower = upper = 0 → ZERO
otherwise         → UNKNOWN
```

Also report:

\[
\Gamma_j
=
\sup_D|\gamma_j|
\]

and, when \(\Gamma_j>0\),

\[
\tau_j^{\mathrm{bound}}
=
\frac{1}{\Gamma_j}.
\]

Label this carefully:

```text
inverse self-feedback magnitude
```

or:

```text
lower bound on a reaction-direction linear timescale
```

It is not a permitted IDAS step size.

For nonsmooth rates:

- retain an absolute bound on the reaction-direction derivative;
- return `UNKNOWN` for the feedback sign unless the active branch is proved.

---

## 5.4 Reaction-interaction matrix

Construct the rate-gradient matrix

\[
R_c
=
\frac{\partial r}{\partial\mathbf c}
\]

and the reaction-space interaction matrix

\[
\boxed{
K=R_cS.
}
\]

Its entries are

\[
K_{jk}
=
\nabla r_j^\mathsf T\nu_k.
\]

Interpretation:

- \(K_{jj}=\gamma_j\): self-feedback;
- \(K_{jk}>0\): advancing reaction \(k\) tends to accelerate reaction \(j\);
- \(K_{jk}<0\): advancing reaction \(k\) tends to suppress reaction \(j\);
- \(K_{jk}=0\): no first-order interaction through the concentration state.

For every structurally nonzero entry, store:

```text
source flux
affected flux
symbolic expression
lower bound
upper bound
absolute upper bound
sign classification
```

Also report:

- structural nonzero count;
- density;
- reaction dependency fan-in and fan-out;
- strongly connected reaction groups;
- one-way coupling between groups;
- largest interaction bounds;
- largest mutually coupled pairs.

Use a small generic strongly-connected-components helper. Do not add a graph-library dependency.

The interaction matrix is representation-dependent, so use source-equivalent fluxes rather than the split directional reactions.

---

## 5.5 Source Jacobian and active stoichiometric modes

### Source Jacobian

Construct

\[
\boxed{
J_c
=
S R_c.
}
\]

Do not perform dense symbolic matrix multiplication.

For each structurally nonzero entry,

\[
(J_c)_{ik}
=
\sum_j
S_{ij}
\frac{\partial r_j}{\partial c_k},
\]

build a sparse unevaluated sum:

```python
sp.Add(*terms, evaluate=False)
```

Calculate:

```text
shape
structural nonzeros
density
row dependency widths
column influence widths
entry bounds
row-sum magnitude bounds
```

### Stoichiometric rank reduction

The complete \(J_c\) contains structural zero modes caused by conserved quantities. Avoid interpreting those as infinitely slow kinetic modes.

Use an exact rank factorisation:

\[
S=B A,
\]

where:

- \(B\in\mathbb R^{n\times q}\) contains independent stoichiometric directions;
- \(A\in\mathbb R^{q\times m}\);
- \(q=\operatorname{rank}(S)\).

For perturbations in the stoichiometric subspace,

\[
\delta\mathbf c=B\delta z,
\]

the reduced Jacobian is

\[
\boxed{
J_{\mathcal S}
=
A R_c B.
}
\]

This matrix describes the active kinetic dynamics after conservation-induced directions have been removed.

Use SymPy’s exact rank decomposition or an equivalent pivot-column factorisation. Store the selected basis directions so the result is reproducible.

Do not calculate exact symbolic eigenvalues.

### State scaling

Raw Jacobian norms depend on variable scaling. Define a deterministic domain scale:

\[
d_i
=
\max\left(
|c_{i,\min}|,
|c_{i,\max}|
\right),
\qquad
D=\operatorname{diag}(d_i).
\]

Then form the similarity-scaled source Jacobian

\[
\widehat J_c
=
D^{-1}J_cD.
\]

This preserves eigenvalues while representing order-one relative concentration perturbations more sensibly.

Call this:

```text
domain-scaled Jacobian
```

not:

```text
IDAS-weighted Jacobian
```

IDA’s actual nonlinear and error tests use weights based on runtime state values together with relative and component-wise absolute tolerances,

\[
W_i=
\frac{1}
{\mathrm{rtol}|y_i|+\mathrm{atol}_i}.
\]

Those weights cannot be reconstructed exactly from `rxn-checker` alone.

### Network magnitude bound

Report:

\[
B_J
=
\sup_D
\|\widehat J_c\|_\infty.
\]

This is a conservative bound on the magnitude of local kinetic linearisation in scaled concentration coordinates.

Also decompose this bound by reaction:

\[
B_j
=
\left\|
D^{-1}
\nu_j
\nabla r_j^\mathsf T
D
\right\|_\infty.
\]

For a rank-one reaction contribution, this is cheaply bounded by

\[
B_j
\leq
\max_i
\frac{|\nu_{ij}|}{d_i}
\sum_k G_{jk}d_k.
\]

Rank the reactions by \(B_j\). This is one of the most actionable outputs:

```text
largest fast-mode contributors
```

### Growth and damping envelope

Calculate an upper bound on the infinity logarithmic norm:

\[
\mu_\infty(\widehat J_c)
=
\max_i
\left(
(\widehat J_c)_{ii}
+
\sum_{k\neq i}
|(\widehat J_c)_{ik}|
\right).
\]

Use signed diagonal bounds where available and absolute bounds otherwise.

Report:

```text
logarithmic_norm_upper
```

Interpretation:

- upper bound \(<0\): certified contraction in the scaled infinity norm;
- upper bound \(>0\): inconclusive, not proof of instability;
- unresolved diagonal derivative: use a conservative absolute replacement and mark the bound as coarse.

Because conservation modes make whole-space contraction uncommon, also calculate the same Gershgorin quantities for the reduced stoichiometric Jacobian \(J_{\mathcal S}\). Label those results as basis-dependent sufficient bounds.

### Spectral-radius envelope

For the reduced Jacobian, use Gershgorin discs to report a conservative bound:

\[
\rho(J_{\mathcal S})
\leq
\max_i
\left(
|c_i|+R_i
\right),
\]

where \(c_i\) is the diagonal centre and \(R_i\) the off-diagonal radius.

Call this:

```text
active-mode magnitude upper bound
```

Do not convert it into a stiffness ratio.

### IDA Newton-shift threshold

For

\[
M_{\mathrm{kin}}(\alpha)
=
\alpha I-\widehat J_c,
\]

a sufficient condition for strict row diagonal dominance is

\[
\alpha
>
\alpha_{\mathrm{dom}},
\]

with

\[
\boxed{
\alpha_{\mathrm{dom}}
=
\max_i
\left(
\overline{J}_{ii}
+
\sum_{k\neq i}
\sup|J_{ik}|
\right).
}
\]

This is numerically the same conservative row bound used for the logarithmic norm.

Report:

```text
IDA alpha diagonal-dominance threshold
```

Interpret it only as:

> For \(\alpha\) above this value, the isolated scaled kinetic Schur complement is certified strictly row diagonally dominant.

Do not convert \(\alpha\) into a step size because the BDF coefficient multiplying \(1/h\) depends on order and step history.

This metric is especially relevant because the current simulation uses `daeIDAS`, supports direct and iterative linear solvers, and defaults to KLU with a small configured nonlinear-iteration budget.  

---

## 5.6 Jacobian-variation and curvature profile

A rate may have a moderate first derivative but a rapidly changing Jacobian. That matters to modified Newton methods because a previously assembled Jacobian may become stale.

IDA’s matrix-based path uses modified Newton iterations and generally reuses a fixed Jacobian within nonlinear iterations, updating it only under selected conditions.

### Sparse Hessian bounds

For each smooth source-equivalent rate, calculate only structurally nonzero second derivatives:

\[
H_{jk\ell}
=
\sup_D
\left|
\frac{\partial^2r_j}
{\partial c_k\,\partial c_\ell}
\right|.
\]

Use symmetry:

\[
H_{jk\ell}=H_{j\ell k}
\]

and calculate only \(k\leq\ell\).

Also collect second derivatives involving temperature or pressure:

\[
\frac{\partial^2r_j}{\partial c_k\partial T},
\qquad
\frac{\partial^2r_j}{\partial c_k\partial P},
\qquad
\frac{\partial^2r_j}{\partial T^2},
\qquad
\frac{\partial^2r_j}{\partial T\partial P},
\qquad
\frac{\partial^2r_j}{\partial P^2}.
\]

Keep concentration–concentration curvature as the primary network result. Report operating-variable curvature separately.

### Source-Jacobian Lipschitz bound

For

\[
F_i=\sum_jS_{ij}r_j,
\]

a conservative concentration-space Jacobian-variation bound is

\[
L_J
=
\max_i
\sum_{j,k,\ell}
|S_{ij}|H_{jk\ell}.
\]

For the scaled coordinates \(c=Dz\), use

\[
\widehat L_J
=
\max_i
\sum_{j,k,\ell}
\frac{|S_{ij}|}{d_i}
H_{jk\ell}d_kd_\ell.
\]

This gives

\[
\|\widehat J(z_1)-\widehat J(z_2)\|_\infty
\leq
\widehat L_J
\|z_1-z_2\|_\infty.
\]

Report:

```text
scaled source-Jacobian variation bound
```

and rank per-reaction contributions:

```text
largest Jacobian-variation contributors
```

### Nonsmooth expressions

If an interior kink is present:

- do not construct a classical global Hessian;
- report `not_applicable_nonsmooth`;
- retain first-order Lipschitz and switch-surface information.

If a kink exists only at the physical boundary:

- report physical-interior curvature if it can be certified;
- report that no closed-domain \(C^{1,1}\) certificate is available;
- inspect the augmented domain separately.

### Work budget

Second derivatives are the only explicitly budgeted part of the profile.

Use a deterministic total budget such as:

```python
_MAX_HESSIAN_ENTRIES = 2048
_MAX_SECOND_DERIVATIVE_OPS = 256
```

The exact constants can be adjusted after running the reforming case.

Rules:

- first derivatives are always attempted;
- second derivatives are attempted only for direct dependencies;
- no simplification is performed before the size check;
- incomplete curvature analysis returns partial results with `truncated=True`;
- exhausting the Hessian budget must never block first-order results.

---

## 6. Operating-variable coupling

Temperature and pressure should not be mixed into the concentration source Jacobian as though they had concentration units.

Construct a separate source-coupling matrix:

\[
J_p
=
S
\begin{bmatrix}
\partial r/\partial T &
\partial r/\partial P
\end{bmatrix}.
\]

Let

\[
d_T=\max(|T_{\min}|,|T_{\max}|),
\qquad
d_P=\max(|P_{\min}|,|P_{\max}|).
\]

Report the scaled columns

\[
\widehat J_{p,T}
=
D^{-1}
\frac{\partial F}{\partial T}
d_T,
\]

\[
\widehat J_{p,P}
=
D^{-1}
\frac{\partial F}{\partial P}
d_P.
\]

These have units of inverse time and represent the source response to an order-one relative change in temperature or pressure.

Report:

```text
temperature coupling bound
pressure coupling bound
most affected species
dominant contributing rates
```

This does not attempt to model the complete feedback loop through the energy, pressure, and closure equations.

---

## 7. Bound semantics

Every reported scalar bound should carry:

```text
value
approximate_value
exact_enclosure
complete
reason
```

Distinguish:

- **exact**: exact affine bound or exact symbolic result;
- **certified enclosure**: conservative interval result;
- **coarse enclosure**: a signed term was replaced by its absolute envelope;
- **partial**: only some entries were bounded;
- **unknown**: no finite bound was established.

Do not hide a coarse \(10^{300}\)-type result behind a normal-looking decimal.

Text should use wording such as:

```text
Certified but very conservative bound: approximately ...
```

Complete exact expressions remain in JSON evidence.

---

## 8. Minimal data model

Use a small set of immutable dataclasses.

```python
@dataclass(frozen=True)
class SurfaceProfile:
    kind: str
    expression: sp.Expr
    source: sp.Expr
    location: SurfaceLocation
    reason: str | None = None


@dataclass(frozen=True)
class DerivativeBound:
    variable: sp.Symbol
    derivative: sp.Expr | None
    lower: sp.Expr | None
    upper: sp.Expr | None
    absolute_upper: sp.Expr | None
    signed: bool


@dataclass(frozen=True)
class RateDifferentialProfile:
    rate_id: str
    regularity: Regularity
    reduced_expression: sp.Expr
    branch_reductions: tuple[Mapping[str, object], ...]
    surfaces: tuple[SurfaceProfile, ...]
    derivatives: tuple[DerivativeBound, ...]
    self_feedback_lower: sp.Expr | None
    self_feedback_upper: sp.Expr | None
    self_feedback_kind: FeedbackKind
    self_feedback_absolute_upper: sp.Expr | None
    source_jacobian_contribution: sp.Expr | None
    curvature_contribution: sp.Expr | None


@dataclass(frozen=True)
class MatrixEnvelope:
    shape: tuple[int, int]
    structural_nonzeros: int
    entries: tuple[Mapping[str, object], ...]
    infinity_norm_upper: sp.Expr | None
    logarithmic_norm_upper: sp.Expr | None
    spectral_radius_upper: sp.Expr | None
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class DomainDifferentialProfile:
    domain: DomainKind
    rates: tuple[RateDifferentialProfile, ...]
    interaction: MatrixEnvelope
    source_jacobian: MatrixEnvelope
    reduced_jacobian: MatrixEnvelope
    operating_coupling: MatrixEnvelope
    ida_alpha_dominance_threshold: sp.Expr | None
    jacobian_variation_upper: sp.Expr | None
    hessian_truncated: bool


@dataclass(frozen=True)
class DifferentialSolverProfile:
    stoichiometric_rank: int
    stoichiometric_basis: tuple[str, ...]
    physical: DomainDifferentialProfile
    augmented: DomainDifferentialProfile
```

Avoid separate classes for:

- Gershgorin discs;
- reaction graphs;
- Hessian tensors;
- scale models;
- solver recommendations.

Those can remain small mappings inside the structured profile.

---

## 9. File layout

Keep the feature contained.

```text
src/rxn_checker/
├── proof/
│   ├── lipschitz.py               # expose per-variable gradient envelope
│   ├── differential.py            # regularity and differential profile
│   └── __init__.py
├── checks/
│   ├── differential_profile.py    # profile → findings/evidence
│   └── core.py                    # registry/profile entry
├── network.py                     # source-equivalent fluxes/rank factorisation
└── context.py                     # optional cached structural objects

tests/
└── test_differential_profile.py
```

Suggested size targets:

```text
proof/differential.py           350–500 lines
lipschitz.py refactor            +40–80 lines
checks/differential_profile.py    70–110 lines
tests                            300–450 lines
```

Do not split this immediately into:

```text
regularity.py
gradient.py
hessian.py
matrix_measures.py
gershgorin.py
solver_scaling.py
```

That would make the implementation harder to follow than the analysis itself.

---

## 10. Shared cached structures

Add only genuinely reusable cached properties to `AnalysisContext`.

```python
@cached_property
def source_equivalent_network(self) -> FluxNetwork:
    return source_equivalent_fluxes(
        self.case.reactions,
        self.stoichiometry,
    )

@cached_property
def stoichiometric_rank_factorization(
    self,
) -> tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]:
    return tuple(
        sp.ImmutableMatrix(item)
        for item in self.source_equivalent_network.stoichiometry.rank_decomposition()
    )
```

Do not cache every Jacobian and Hessian as a dense SymPy matrix in `AnalysisContext`.

Differential expressions depend on:

- domain-specific branch reduction;
- physical versus augmented analysis;
- whether a signed or absolute derivative is requested.

Those caches should remain private to `profile_differential`.

---

## 11. Pure proof-layer API

Expose one public function:

```python
def profile_differential(
    *,
    analyzer: ExpressionAnalyzer,
    reactions: Sequence[Reaction],
    stoichiometry: sp.MatrixBase,
    concentration_symbols: Sequence[sp.Symbol],
    operating_symbols: Sequence[sp.Symbol],
    physical_domain: ConcentrationDomain,
    augmented_domain: ConcentrationDomain,
) -> DifferentialSolverProfile:
    ...
```

Internally:

```python
def _profile_domain(...): ...
def _reduce_branches(...): ...
def _extract_surfaces(...): ...
def _classify_surface(...): ...
def _profile_rate(...): ...
def _build_rate_gradient(...): ...
def _build_interaction(...): ...
def _build_source_jacobian(...): ...
def _rank_reduced_jacobian(...): ...
def _matrix_envelope(...): ...
def _profile_curvature(...): ...
```

Prefer top-level functions over one large function containing many closures.

Use sparse dictionaries:

```python
dict[tuple[int, int], sp.Expr]
```

for derivative and matrix entries. Convert to SymPy matrices only where exact rank factorisation requires one.

---

## 12. Main algorithm

The top-level flow should remain easy to read:

```python
def profile_differential(...):
    flux_network = source_equivalent_fluxes(reactions, stoichiometry)
    basis, coordinates = rank_factorization(flux_network.stoichiometry)

    physical = _profile_domain(
        analyzer,
        flux_network,
        concentration_symbols,
        operating_symbols,
        physical_domain,
        basis,
        coordinates,
    )
    augmented = _profile_domain(
        analyzer,
        flux_network,
        concentration_symbols,
        operating_symbols,
        augmented_domain,
        basis,
        coordinates,
    )
    return DifferentialSolverProfile(
        stoichiometric_rank=basis.cols,
        stoichiometric_basis=flux_network.basis_ids,
        physical=physical,
        augmented=augmented,
    )
```

Within one domain:

```python
def _profile_domain(...):
    rates = tuple(
        _profile_rate(...)
        for flux in flux_network.fluxes
    )
    gradient = _collect_sparse_gradient(rates)
    interaction = _build_interaction(gradient, stoichiometry)
    source = _build_source_jacobian(stoichiometry, gradient)
    reduced = _reduce_source_jacobian(
        gradient,
        basis,
        coordinates,
    )
    operating = _build_operating_coupling(...)
    curvature = _profile_curvature(...)
    return DomainDifferentialProfile(...)
```

The implementation should read in the same order as the mathematical description.

---

## 13. Check registration

Register one case-scoped, nonblocking analysis:

```python
_spec(
    "differential_solver_profile",
    "Differential solver profile",
    Stage.ANALYSIS,
    CheckScope.CASE,
    (),
    run_differential_solver_profile,
    blocking=False,
)
```

Include it in:

```text
analysis
all
```

Do not include it in:

```text
basic
physical
robust
```

unless the user explicitly requests it.

The check verdict should mean whether the analysis completed:

- `PASS`: all first-order core results were obtained;
- `UNKNOWN`: one or more essential first-order sections are unresolved;
- never `FAIL` merely because a rate is nonsmooth, strongly coupled, or self-amplifying.

The profile remains `Role.ANALYSIS`, so its verdict does not affect the overall case verdict.

---

## 14. Text report

Keep the text result selective and place full matrices in JSON.

Illustrative structure:

```text
PASS     Differential solver profile

         Kinetic source rank 4; 6 source-equivalent fluxes.
         Physical domain: 4 C1 rates, 2 piecewise-C1 rates;
         3 boundary switch surfaces and no proved interior singularities.

         Domain-scaled source-Jacobian bound: ...
         Active-mode magnitude upper bound: ...
         IDA alpha diagonal-dominance threshold: ...
         Source-Jacobian variation bound: ...

         Strongest self-feedback:
           xu_froment.smr: damping, gamma in [...]
           medrano.reduction_co: mixed, gamma in [...]

         Largest fast-mode contributors:
           medrano.reduction_co: ...
           xu_froment.smr: ...
           medrano.reduction_h2: ...

         Augmented domain: 4 additional active switch surfaces;
         signed feedback unresolved for 2/6 fluxes.
```

Only show:

- domain-level regularity counts;
- source rank;
- primary Jacobian metrics;
- three strongest feedback magnitudes;
- three largest Jacobian contributors;
- significant physical-to-augmented changes;
- incomplete sections.

Do not print the full derivative or interaction matrices in text.

---

## 15. JSON evidence

Use one evidence object:

```json
{
  "target": "kinetic concentration-space subsystem",
  "solver_context": {
    "form": "F(t, y, y_dot) = 0",
    "kinetic_schur_complement": "alpha*I - S*dr_dc",
    "full_daetools_jacobian": false
  },
  "network": {
    "declared_reactions": 9,
    "source_equivalent_fluxes": 6,
    "stoichiometric_rank": 4,
    "stoichiometric_basis": []
  },
  "physical": {
    "regularity_summary": {},
    "rates": {},
    "interaction_matrix": {},
    "source_jacobian": {},
    "reduced_jacobian": {},
    "operating_coupling": {},
    "ida_alpha_dominance_threshold": null,
    "jacobian_variation_upper": null
  },
  "augmented": {
    "regularity_summary": {},
    "rates": {},
    "interaction_matrix": {},
    "source_jacobian": {},
    "reduced_jacobian": {},
    "operating_coupling": {},
    "ida_alpha_dominance_threshold": null,
    "jacobian_variation_upper": null
  }
}
```

For sparse matrices, serialize entries as:

```json
{
  "row": "CH4",
  "column": "H2",
  "expression": "...",
  "lower": "...",
  "upper": "...",
  "absolute_upper": "...",
  "signed": true
}
```

The complete sparse matrices may be retained in JSON. Avoid serializing dense arrays full of zeros.

---

## 16. Tests

### 16.1 Linear decay

For

\[
A\rightarrow B,
\qquad
r=kA,
\]

verify:

\[
\gamma=-k,
\]

and that the reaction is classified as uniformly damping.

Verify the source Jacobian:

\[
J=
\begin{bmatrix}
-k&0\\
k&0
\end{bmatrix}.
\]

The reduced stoichiometric Jacobian should be the scalar \(-k\).

### 16.2 Autocatalytic reaction

For

\[
A+B\rightarrow2B,
\qquad
r=kAB,
\]

verify:

\[
\gamma=k(A-B).
\]

Use a domain where:

- \(A>B\) always, giving `AMPLIFYING`;
- \(A<B\) always, giving `DAMPING`;
- both cases occur, giving `MIXED`.

### 16.3 Reversible grouping

For:

```text
A → B, rate kf*A
B → A, rate kb*B
```

verify that the source-equivalent system produces:

\[
r_{\mathrm{net}}=k_fA-k_bB
\]

and that its source Jacobian equals the declared two-rate source Jacobian exactly.

### 16.4 Conservation-mode removal

For a mass-conserving two-species reaction, confirm:

- full \(J_c\) has a structural zero mode;
- stoichiometric rank is one;
- the reduced Jacobian contains only the active kinetic mode.

### 16.5 Reaction interaction

Use two coupled reactions:

```text
A → B
B → C
```

and verify:

- the first reaction changes the second reaction’s rate when \(r_2\) depends on \(B\);
- the reverse coupling is absent when \(r_1\) depends only on \(A\);
- the graph contains a one-way edge.

### 16.6 Branch reduction

Test:

```python
sp.Max(A, 0)
sp.Abs(A)
sp.Min(1, A)
```

on domains where the relevant branch is:

- always active;
- boundary-only;
- crossed in the interior;
- unresolved.

Confirm that derivatives are taken after proven branch reduction.

### 16.7 Physical versus augmented domain

Use:

\[
r=\max(A,0)B.
\]

Confirm:

- physical domain \(A\geq0\): expression reduces to \(AB\);
- augmented domain crossing \(A=0\): profile is piecewise \(C^1\);
- physical signed feedback is available;
- augmented signed feedback becomes unresolved or branch-dependent.

### 16.8 Smooth but sharp gate

For:

\[
g_\varepsilon(A)=\frac{A}{A+\varepsilon},
\]

verify:

\[
g_\varepsilon'(0)=\frac1\varepsilon,
\qquad
|g_\varepsilon''(0)|=\frac{2}{\varepsilon^2}.
\]

Confirm that:

- regularity is smooth;
- first-derivative and curvature bounds increase as \(\varepsilon\) decreases;
- no nonsmooth surface is reported.

### 16.9 Derivative singularity

Use:

\[
r=\sqrt A
\]

on \(A\in[0,1]\).

Confirm:

- rate is continuous;
- derivative is singular at the boundary;
- no finite closed-domain gradient bound is reported;
- a strictly positive lower bound on \(A\) restores a finite derivative bound.

### 16.10 Temperature and pressure coupling

Use:

\[
r=A\exp\left(-\frac{E}{RT}\right)P.
\]

Verify separate concentration, temperature, and pressure derivative bounds and the scaled operating-coupling columns.

### 16.11 Jacobian variation

For:

\[
r=kA^2,
\]

verify:

\[
\frac{\partial r}{\partial A}=2kA,
\qquad
\frac{\partial^2r}{\partial A^2}=2k.
\]

Check the source-Jacobian variation bound exactly.

### 16.12 IDA shift threshold

For a small exact Jacobian, manually calculate:

\[
\alpha_{\mathrm{dom}}
=
\max_i
\left(
\overline J_{ii}+
\sum_{k\neq i}|J_{ik}|
\right)
\]

and verify the implementation.

### 16.13 Unsupported derivative

Use a custom SymPy function and confirm:

- structural dependency is retained;
- the unsupported derivative is identified;
- other rates still receive complete profiles;
- check verdict becomes `UNKNOWN`;
- the analysis does not raise.

### 16.14 Reforming regression

Use stable structural assertions rather than exact giant constants:

```text
9 declared directional reactions
6 source-equivalent fluxes
stoichiometric rank 4
nonzero concentration and temperature derivatives
physical and augmented profiles both returned
at least one physical-domain branch reduction
more active switches on augmented than physical domain
source Jacobian is sparse
reaction interaction matrix is sparse
Hessian budget is not exceeded, or truncation is reported deterministically
no exact symbolic eigenvalue calculation occurs
```

Also assert that the largest reported contributors correspond to actual rate IDs and not generated temporary symbols.

---

## 17. Implementation sequence

### Step 1 — Shared source-equivalent network

Implement or reuse:

```text
proportional stoichiometric-column grouping
source-equivalent flux expressions
exact rank factorisation
deterministic basis IDs
```

This should be shared with `evaluation_profile`.

### Step 2 — Generalise the Lipschitz traversal

Change the recursive Lipschitz machinery to retain per-variable absolute derivative envelopes.

Confirm that all existing Lipschitz tests and reported constants remain unchanged.

### Step 3 — Branch reduction and regularity surfaces

Implement:

```text
domain-aware Abs/Min/Max reduction
surface extraction
surface location classification
regularity classification
```

Test this independently before constructing network matrices.

### Step 4 — Signed derivatives and self-feedback

Add:

```text
cached sparse symbolic derivatives
signed derivative bounds
reaction-direction derivative
feedback classification
per-flux timescale-magnitude bound
```

### Step 5 — Interaction and source Jacobians

Build sparse:

```text
R_c
R_c*S
S*R_c
reduced stoichiometric Jacobian
operating-variable source coupling
```

Add structural graphs, SCCs, and contributor rankings.

### Step 6 — Scaling and IDA-facing metrics

Add:

```text
domain scaling
scaled infinity-norm bound
logarithmic-norm upper bound
Gershgorin active-mode bound
IDA alpha diagonal-dominance threshold
```

### Step 7 — Curvature

Add budgeted:

```text
sparse Hessian entries
scaled source-Jacobian variation
per-reaction curvature contributions
physical/augmented comparison
```

### Step 8 — Check and reporting integration

Add:

```text
check registry entry
analysis/all profile selection
concise text summary
complete sparse JSON evidence
reforming regression test
```

---

## 18. Deliberate non-goals

Do not include:

- an actual IDAS step-size estimate;
- a scalar “solver friendliness” score;
- a stiffness ratio based on global worst-case bounds;
- exact symbolic eigenvalues;
- numerical eigenvalue sampling;
- random or grid sampling of the state domain;
- the transport Jacobian;
- the full spatially discretised DAE;
- linear-solver fill-in estimates;
- Jacobian condition numbers;
- DAETools imports;
- automatic modification of reaction expressions;
- automatic selection of smoothing widths;
- automatic replacement of `Max`, `Min`, or `Abs`;
- runtime claims such as “this reaction adds 20% simulation time.”

The existing `multisolid-CL` incidence-matrix export can later be used to compare symbolic kinetic sparsity with the initialized full DAETools structure, but that validation should remain outside `rxn-checker`. 

---

## 19. Completion criteria

The profile is complete when it can answer, without running DAETools:

1. Which rate expressions are \(C^1\), piecewise \(C^1\), or only Lipschitz?
2. Which switch or singular surfaces intersect the physical or augmented domain?
3. Which branches are redundant on the physical domain?
4. Which state variable dominates each rate’s gradient?
5. How sensitive are the source terms to temperature and pressure?
6. Is each source-equivalent reaction direction self-damping, self-amplifying, mixed, or unresolved?
7. Which reaction pairs are strongly coupled?
8. What is the sparse kinetic source Jacobian?
9. What active kinetic modes remain after conserved directions are removed?
10. Which reactions dominate the source-Jacobian magnitude?
11. What conservative magnitude and growth bounds can be certified?
12. How large must the IDA shift \(\alpha\) be for a sufficient kinetic diagonal-dominance certificate?
13. How rapidly can the source Jacobian vary?
14. Which rates make a reused Jacobian most likely to become stale?
15. Which conclusions change when moving from the physical to the augmented domain?

That gives the reaction author a useful solver-facing diagnosis while keeping the implementation symbolic, modular, and substantially smaller than an attempted model of the complete DAETools solver.