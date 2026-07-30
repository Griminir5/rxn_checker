# Case Specification
 - Each case specifies a list of species
 - Each case specifies a list of reactions
 - Optional domain overrides
 - Lower bound perturbation magnitude
 - State variables are molar concentration of specified components, temperature, pressure

# Reaction Definition
 - Reaction implementation are on the user
 - Must expose a 'Reaction' object that documents id, family, rate expression. stoichiometry, catalysts
 - Expression is symbolic, constructed with SymPy

# Basic Checks
 - All requested species and reactions exist
 - All required reactants, products, catalysts present in species
 - Atom conservation
 - Mass conservation

# Physical Checks
 - Rates are non-negative in physical domain
 - Rate is 0 when reactant/catalyst concentration is 0
 - Full network positivity
 - Locate equilibria and terminal faces
 - Report Jacobian eigenvalues within the stoichiometric subspace
 - Test recovery from nonphysical domain
 - Detect unbounded growth
 - Detect finite-time blow-up
 - Detect bound violations
 - Detect unintended stable equilibria

# Numeric and Solver Friendliness Checks
 - Detect improper forms (fractional powers, bad logarithms, divisions)
 - Check for NaNs, infs, exceptions in extended domain
 - Minimum denominator margin
 - Scaled rate and full-source Jacobians
 - Stiffness, eigenvalue spread
 - Finite difference Jacobian stability
 - Experssion graph metrics (op count, intermediate-value dynamic range, cancellation factors, graph perturbation amplification)
 - Time Jacobian Evaluation
 - Time rate evaluation
 - Rate mismatch
 - Jacobian mismatch
 - Equilibrium mismatch
 - Integration runtime ratio
