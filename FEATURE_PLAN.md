# Roadmap

The implemented case, reaction, registry, and conservation-check foundations
are documented in the README. Remaining work is grouped below.

## Case configuration

- Optional physical-domain overrides
- Lower-bound perturbation magnitude

## Physical checks

- Bounded rate non-negativity using interval branch-and-bound
- Network positivity, equilibria, and terminal faces
- Jacobian eigenvalues within the stoichiometric subspace
- Recovery from the nonphysical domain and bound violations
- Unbounded growth, finite-time blow-up, and unintended stable equilibria

## Numerical robustness

- Unsafe fractional powers, logarithms, and divisions
- NaNs, infinities, and exceptions in the extended domain
- Denominator margins and scaled rate/source Jacobians
- Stiffness, eigenvalue spread, and finite-difference Jacobian stability
- Expression-graph size, dynamic range, cancellation, and perturbation metrics
- Rate and Jacobian evaluation time
- Rate, Jacobian, equilibrium, and integration-runtime comparisons
