# Symbolic Recovery from Nonphysical Concentrations

## Goal

Numerical solvers can produce small negative concentrations even when the true solution is nonnegative. This check asks:

> If one or more concentrations become slightly negative, does the reaction system push the error back toward the physical domain, leave it stuck, or make it worse?

The check is entirely symbolic. It needs no initial state, simulation, sampling, or numerical search.

It is a **system-level** check on

$$
\dot c = F(c) = S\,r(c),
$$

where:

- $c\in\mathbb R^n$ is the concentration vector;
- $r(c)\in\mathbb R^m$ is the vector of reaction rates;
- $S\in\mathbb R^{n\times m}$ is the stoichiometric matrix;
- $F(c)$ is the net production or consumption rate of every species.

Checking individual reaction rates is not enough because several reactions and several negative species can interact.

## 1. First ask whether recovery is possible

Reaction kinetics cannot repair every arbitrary negative state because reactions preserve conserved totals.

For example, for

$$
A\rightarrow B,
$$

$A+B$ is conserved.

- $A=-0.1,\ B=1$ is potentially repairable because $A+B=0.9$, and the same compatibility class contains the physical state $A=0,\ B=0.9$.
- $A=-1,\ B=0.2$ is not repairable because $A+B=-0.8$, while two nonnegative concentrations cannot have a negative sum.

A state $c$ is **stoichiometrically repairable** when its stoichiometric compatibility class contains at least one nonnegative state:

$$
\left(c+\operatorname{im}S\right)\cap\mathbb R_+^n\neq\varnothing.
$$

Define the cone of nonnegative conserved quantities:

$$
\mathcal Q=\left\{q:q\geq0,\ q^\top S=0\right\}.
$$

By Farkas' lemma, $c$ is repairable exactly when

$$
q^\top c\geq0
\qquad\text{for every }q\in\mathcal Q.
$$

It is sufficient to test the extreme rays of $\mathcal Q$, which can be obtained from the existing conserved-quantity machinery.

Consequences:

- A negative inert species cannot be repaired by reaction kinetics.
- A state with negative conserved mass, atom inventory, or site inventory cannot be repaired.
- These cases should be reported as **STOICHIOMETRICALLY_UNREPAIRABLE**, not as failures of the kinetic law.

## 2. Componentwise recovery

For species $i$,

$$
\dot c_i=F_i(c).
$$

If $c_i<0$, then $F_i(c)>0$ moves it upward toward zero. The strongest simple recovery condition is therefore

$$
c_i<0\quad\Longrightarrow\quad F_i(c)>0
$$

for every currently negative species.

This is **componentwise recovery**: every negative concentration immediately improves. It is easy to interpret, but stronger than necessary because one species could temporarily worsen while the total negative error still decreases.

## 3. Net recovery

Let $\delta_i>0$ be the permitted negative excursion for species $i$. Define a normalized negative deficit

$$
d_i(c)=\frac{\max(-c_i,0)}{\delta_i}
$$

and the total nonphysical error

$$
V(c)=\frac12\sum_i d_i(c)^2.
$$

$V=0$ exactly when every concentration is nonnegative. In a region where the set of negative species is $N$,

$$
\dot V(c)
=\sum_{i\in N}\frac{c_iF_i(c)}{\delta_i^2}.
$$

Equivalently, define the restoration score

$$
R_N(c)
=\sum_{i\in N}\frac{(-c_i)F_i(c)}{\delta_i^2}
=-\dot V(c).
$$

The signs have a simple interpretation:

| Classification | Symbolic condition | Meaning |
|---|---:|---|
| Strongly restoring | $F_i>0$ for every $i\in N$ | Every negative species improves |
| Net restoring | $R_N>0$ | Total negative error decreases |
| Non-worsening | $R_N\geq0$ | Total negative error cannot increase |
| Stuck | $R_N=0$ on a nonphysical invariant set | The error remains negative |
| Worsening | $R_N<0$ somewhere feasible | Total negative error grows there |

Componentwise recovery implies net recovery, but not conversely. The main pass condition should be net recovery; componentwise recovery is a stronger diagnostic.

## 4. Why simultaneous negative species matter

Consider

$$
A+B\rightarrow C,
\qquad r=kAB,
\qquad k>0.
$$

If only $A<0$ and $B>0$, then $r<0$ and

$$
\dot A=-r>0.
$$

Thus $A$ recovers. The same singleton test succeeds for $B$.

If both $A<0$ and $B<0$, however, then $AB>0$, so

$$
r>0,\qquad
\dot A=-r<0,\qquad
\dot B=-r<0.
$$

Both concentrations become more negative. The checker must therefore examine every feasible nonempty negative-species set $N$, not merely one species at a time.

The exponential search can be reduced by analysing independent stoichiometric connected components separately.

## 5. Keep the state inside the checked excursion band

Suppose the symbolic guarantee only covers

$$
c_i\geq-\delta_i.
$$

On the lower boundary, require

$$
c_i=-\delta_i
\quad\Longrightarrow\quad
F_i(c)\geq0.
$$

This prevents the reaction source from pushing species $i$ outside the region in which recovery was certified.

The related physical-boundary condition is

$$
c_i=0
\quad\Longrightarrow\quad
F_i(c)\geq0.
$$

The two properties answer different questions:

- **Physical-domain invariance:** can a physical trajectory be pushed negative?
- **Nonphysical recovery:** once slightly negative, is it pushed back toward the physical domain?

## 6. Do not require nonnegative rates outside the physical domain

For the directional reaction $A\rightarrow B$, let

$$
r=kA,\qquad k>0.
$$

When $A<0$, the extended rate is negative, but

$$
\dot A=-r=-kA>0,
$$

so $A$ is restored. Requiring the directional rate itself to remain nonnegative outside the physical domain would reject useful recovery behaviour.

| Extension for $A\rightarrow B$ | Behaviour when $A<0$ |
|---|---|
| $r=kA$ | Restoring |
| $r=k\max(A,0)$ | Stuck |
| $r=k\lvert A\rvert$ | Worsening |
| $r=k\sqrt A$ | Undefined over the reals |

Rate nonnegativity should be checked on the physical domain. In the negative extension, the relevant object is the network source $F=Sr$.

## 7. Proposed certificate

For every stoichiometrically repairable nonphysical state in the declared excursion domain, prove:

1. All rates and source terms are real and defined.
2. $R_N(c)>0$, so the total negative error strictly decreases.
3. On every lower excursion face $c_i=-\delta_i$, $F_i(c)\geq0$.

Also report whether the stronger componentwise conditions $F_i(c)>0$ hold for all negative species.

The precise claim is:

> Throughout the declared, stoichiometrically repairable negative-extension region, the reaction source is defined, cannot push the state through a lower excursion boundary, and strictly decreases the chosen measure of negative-concentration error.

This wording avoids claiming more than was proved. Strict decrease of $V$ is a restoring certificate; a formal claim of global asymptotic attraction additionally needs the usual dynamical assumptions, such as existence of solutions and bounded forward trajectories.

## 8. Symbolic implementation

Construct $F=Sr$ once as shared network data. Compute the extreme rays of the nonnegative conservation cone.

For each nonempty candidate negative set $N$:

1. Create the sign-region constraints:
   - $-\delta_i\leq c_i<0$ for $i\in N$;
   - $c_i\geq0$ for $i\notin N$;
   - all configured symbolic bounds and parameter assumptions.
2. Add $q_k^\top c\geq0$ for every nonnegative conservation ray $q_k$.
3. Prove the constrained region infeasible, or continue.
4. Prove that every rate and source term is real and defined.
5. Construct $R_N$ and prove its sign.
6. Separately prove the signs of $F_i$ for all $i\in N$.
7. Substitute $c_i=-\delta_i$ and prove $F_i\geq0$ on each lower excursion face.

~~~python
F = stoichiometric_matrix @ rate_vector
q_rays = nonnegative_conservation_rays(stoichiometric_matrix)

for negative_species in candidate_negative_sets():
    constraints = sign_region(negative_species)
    constraints += parameter_assumptions()
    constraints += [q.dot(c) >= 0 for q in q_rays]

    if prove_infeasible(constraints):
        continue

    require_real_and_defined(F, constraints)

    restoration = sum(
        (-c[i]) * F[i] / excursion_size[i]**2
        for i in negative_species
    )

    net_result = prove_sign(restoration, constraints)
    component_results = {
        i: prove_sign(F[i], constraints)
        for i in negative_species
    }
    lower_face_results = {
        i: prove_nonnegative(
            F[i].subs(c[i], -excursion_size[i]),
            constraints,
        )
        for i in negative_species
    }
~~~

The proof pipeline should remain exact:

- substitute positive dummy variables to encode sign-regions where useful;
- apply declared parameter assumptions explicitly;
- refine <code>Abs</code>, <code>Max</code>, <code>Min</code>, and <code>Piecewise</code>;
- separate numerator and denominator signs;
- recognize products and sums of sign-known factors;
- use exact polynomial or rational feasibility/positivity methods where applicable;
- return **INDETERMINATE** when no proof or exact symbolic counterexample is available.

Numerical sampling must not be presented as proof and is outside this check's scope.

## 9. Verdicts

- **STRONGLY_RESTORING:** every negative component satisfies $F_i>0$.
- **NET_RESTORING:** $R_N>0$, but componentwise recovery was not proved.
- **NON_WORSENING:** only $R_N\geq0$ was proved.
- **STUCK:** a nonphysical invariant state or region with $R_N=0$ was proved.
- **WORSENING:** $R_N<0$ was proved on a feasible region.
- **UNDEFINED_IN_EXTENSION:** a rate or source is not real and finite.
- **STOICHIOMETRICALLY_UNREPAIRABLE:** the compatibility class does not intersect the nonnegative orthant; this is not a kinetics failure.
- **INDETERMINATE:** the symbolic engine could not prove a classification.

Each result should identify:

- the negative-species set;
- the assumptions used;
- the decisive simplified expression;
- any exact symbolic counterexample or counterexample conditions.

## 10. Recovery does not necessarily mean finite-time re-entry

For $A\rightarrow B$ with $r=kA$,

$$
A(t)=A(0)e^{-kt}.
$$

Starting from $A(0)<0$, the concentration approaches zero from below but never crosses it in finite continuous time. The kinetics are restoring, but finite-time re-entry has not been proved.

The report should therefore distinguish:

- **Restoring/attracting:** negative error decreases toward zero.
- **Finite re-entry:** the trajectory is proved to cross into the nonnegative domain in finite time.
- **Stuck:** the negative error does not shrink.
- **Worsening:** the negative error grows.

Finite re-entry needs a stronger condition, normally a uniform inward rate near the physical boundary. Exact nonnegative solver iterates are ultimately also an integrator or projection concern, not solely a property of the kinetic expression.
