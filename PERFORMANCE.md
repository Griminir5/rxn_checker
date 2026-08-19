# Runtime analysis

This document records the main scaling costs found in the symbolic checks and
the invariants preserved by the runtime work.

## Measured cases

Wall-clock measurements use CPython 3.11 and SymPy 1.14 in the repository
virtual environment. They are intended as relative measurements, not stable
performance contracts.

| Workload | Before | After |
|---|---:|---:|
| Reforming enabled checks, recovery disabled | about 96 s | about 26 s |
| Reforming complete report, recovery enabled | did not finish in the profiling window | about 39 s |
| Example complete report, recovery enabled | not applicable | about 0.22 s |
| 100 independent decay species, terminal faces | powerset is intractable | about 0.9 s / 200 face tests |

The reforming recovery check now returns an exact `FAIL` certificate after four
regions. Its registered runner takes about 12 seconds. Calling
`check_nonphysical_recovery()` directly still requests the full bounded region
analysis unless `stop_on_failure=True` is supplied.

## Removed hot paths

- Rate sign candidates are lazy. `factor_terms()` and `factor()` are no longer
  evaluated when the original assumed expression already has a known sign.
- Linear-domain feasibility is sent directly to SymPy's exact simplex matrix
  interface. This avoids converting every affine relation through univariate
  set solving. No floating-point LP or sampled certificate is used.
- Float-to-rational rebuilding is delayed until an affine expression actually
  enters the exact LP. Large nonlinear rate DAGs are no longer reconstructed
  merely to ask realness or sign questions.
- Recovery definedness traverses rate laws, not every expanded source sum. A
  finite real linear combination of finite real rates is necessarily finite
  and real, so the removed source traversal was logically redundant.
- Registered recovery stops after an exact failure or boundary-violation
  witness. It never stops early on a possible pass.
- The independent negative-side check constructs only two restricted domains
  per eligible species (`x_i <= 0` and `x_i < 0`). It therefore scales
  linearly in the number of species rather than enumerating negative-species
  subsets, and returns `INDETERMINATE` above its source-operation limit.
- Terminal faces use dependency-directed search, cached one-symbol
  restrictions, cached zero proofs, and verified reactant/catalyst zero
  boundaries. Cancellation between reactions is still checked on the exact
  source expression.
- Stoichiometry, rates, source terms, and conservation results are cached in the
  per-run `CheckContext` instead of being rebuilt by several checks.
- Conservation-cone rays use exact incremental double description. The new
  implementation was compared with the former exhaustive support algorithm on
  2,800 randomized small integer matrices with identical ray sets.

## Remaining worst cases

Some output sets are intrinsically exponential. A network can have
exponentially many conservation rays or maximal coordinate faces, and the full
recovery API explicitly returns one result per negative-species region. The
implementation is now output/dependency directed and has conservative work
limits, but no exact general algorithm can materialize exponentially many
distinct results in polynomial time.

For a future complete recovery `PASS` on one dense 100-species stoichiometric
component, the result representation will need a compressed global certificate
(for example, symbolic sign-pattern clauses) instead of a tuple containing up
to `2**100 - 1` region records. Failure detection, sparse face discovery, and
networks split into small components already avoid that powerset.
