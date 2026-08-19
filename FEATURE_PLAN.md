# Roadmap

The implemented case, reaction, registry, and conservation-check foundations
are documented in the README. Remaining work is grouped below.

## Case configuration

- Optional physical-domain overrides
- Lower-bound perturbation magnitude

## Physical checks

- Jacobian eigenvalues within the stoichiometric subspace

## Numerical robustness

- Denominator margins and scaled rate/source Jacobians
- Stiffness, eigenvalue spread, and finite-difference Jacobian stability
- Expression-graph size, dynamic range, cancellation, and perturbation metrics
- Rate and Jacobian evaluation time
- Rate, Jacobian, equilibrium, and integration-runtime comparisons
