# rxn-checker

`rxn-checker` supplies the import-safe objects underneath checks for whether
reaction expressions and reaction networks are friendly to numerical solvers.
For example, a later checker can ask whether a reaction pushes a tiny negative
component concentration back toward the physical region or farther away from
it.

This package currently defines the integration seam. It does not assume that
the consumer is building an ODE: it evaluates one-way reaction implementations
independently and retains their expressions, full stoichiometry, and
provenance.

## Core building blocks

- `Stoichiometry` preserves separate reactant, product, and catalyst roles.
- `ReactionContext` is the narrow interface through which an implementation
  accesses registered states and parameters.
- `Reaction` holds an id, full stoichiometry, declared dependencies, and one
  callable forward implementation.
- `Case` registers state expressions and parameter values, selects reactions,
  validates their requirements, and invokes their hooks.
- `EvaluatedReaction` is the neutral output: one expression tied to its
  reaction metadata. It can report a net stoichiometry-weighted contribution
  for a component, but it does not assemble an equation system.

This follows the same boundary used by `multisolid-CL`: metadata remains safe to
inspect, implementations receive a context instead of importing the enclosing
model, and the general object resolves the hooks.

## Full, sided stoichiometry

Net coefficients alone are insufficient. Both of these reactions have the net
change `A: +1, B: -1`, but their mechanisms are not the same:

```text
A + B -> 2 A
    B -> A
```

They remain distinguishable in the core model:

```python
from rxn_checker import Stoichiometry

autocatalytic = Stoichiometry(
    reactants={"A": 1, "B": 1},
    products={"A": 2},
)
simple = Stoichiometry(
    reactants={"B": 1},
    products={"A": 1},
)

assert autocatalytic != simple
assert dict(autocatalytic.net) == {"A": 1, "B": -1}
assert dict(simple.net) == {"B": -1, "A": 1}
```

`net` and `net_coefficient(state_id)` are derived conveniences. The reactant and
product mappings remain the authoritative description.

A conserved participant may be written on both sides:

```python
enzyme_reaction = Stoichiometry(
    reactants={"enzyme": 1, "A": 1},
    products={"enzyme": 1, "B": 1},
)
```

Alternatively, a catalytic component which is important to the expression but
does not belong on either side can be recorded explicitly:

```python
surface_reaction = Stoichiometry(
    reactants={"A": 1},
    products={"B": 1},
    catalysts={"surface_site": 2},
)
```

Reactant, product, and catalyst coefficients must be finite positive constants.
Explicit catalysts must not also appear in a reactant or product mapping; use
the both-sides form when that is the intended description.

## Defining and hooking in a reaction

```python
from rxn_checker import (
    Case,
    Reaction,
    ReactionContext,
    Stoichiometry,
    parameter,
    state,
)


def autocatalytic_rate(context: ReactionContext):
    """A complete symbolic implementation supplied by a user."""
    A = context.state("A")
    B = context.state("B")
    k = context.parameter("k")
    return k * A * B


reaction = Reaction(
    id="autocatalytic_conversion",
    name="A-catalysed conversion of B to A",
    stoichiometry=Stoichiometry(
        reactants={"A": 1, "B": 1},
        products={"A": 2},
    ),
    parameter_dependencies=("k",),
    implementation=autocatalytic_rate,
    source_reference="example",
)

A = state("A")
B = state("B")
k = parameter("k", positive=True)

symbolic_case = Case(
    name="symbolic",
    states={"A": A, "B": B},
    parameters={"k": k},
    reactions=(reaction,),
)

evaluated = symbolic_case.evaluate_reaction("autocatalytic_conversion")
assert evaluated.expression == k * A * B
assert dict(evaluated.stoichiometry.reactants) == {"A": 1, "B": 1}
assert dict(evaluated.stoichiometry.products) == {"A": 2}
assert evaluated.contribution("A") == k * A * B
assert evaluated.contribution("B") == -k * A * B
```

Parameter entries are values, not necessarily symbols, so the same reaction can
be attached to a concrete case without changing its implementation:

```python
concrete_case = Case(
    states={"A": A, "B": B},
    parameters={"k": 2.0},
    reactions=(reaction,),
)

assert concrete_case.expressions["autocatalytic_conversion"] == 2.0 * A * B
```

Callable objects work as implementations too. This lets a larger user-owned
reaction object carry correlations or supporting methods while presenting the
same single `__call__(context)` hook.

## Dependencies and catalysts

Every reactant, product, and catalyst in the stoichiometry is automatically
visible to the hook. Other modifiers—temperature or an inhibitor, for
example—must be listed with `state_dependencies`:

```python
def inhibited(context: ReactionContext):
    return (
        context.parameter("k")
        * context.state("A")
        * context.state("surface_site")
        / (1 + context.state("inhibitor"))
    )


inhibited_reaction = Reaction(
    id="inhibited_loss",
    stoichiometry=Stoichiometry(
        reactants={"A": 1},
        products={},
        catalysts={"surface_site": 1},
    ),
    state_dependencies=("inhibitor",),
    parameter_dependencies=("k",),
    implementation=inhibited,
)
```

A hook receives only its declared dependencies. Missing case entries are
reported when the case is created; attempts to access undeclared entries are
reported when the hook runs. The returned expression is also checked for
closed-over symbols outside the declared dependency set.

## Reactions are one-way

`Reaction` always describes one forward direction. There is no reversible flag
and no forward-minus-reverse implementation. When both directions are needed,
they are two reactions with separate stoichiometry and separate hooks:

```python
A_to_B = Reaction(
    id="A_to_B",
    stoichiometry=Stoichiometry(reactants={"A": 1}, products={"B": 1}),
    implementation=lambda context: context.state("A"),
)
B_to_A = Reaction(
    id="B_to_A",
    stoichiometry=Stoichiometry(reactants={"B": 1}, products={"A": 1}),
    implementation=lambda context: context.state("B"),
)
```

This does not require an expression to remain positive outside the physical
domain. Whether a nominally forward expression changes sign after a solver
produces a negative concentration is exactly the sort of property a later
checker should report.

## Why states are not assumed non-negative

`state("A")` creates a real SymPy symbol but deliberately does not give it a
non-negative assumption. A non-negative assumption could simplify away exactly
the behaviour the checker needs to inspect.

```python
quadratic_loss = Reaction(
    id="quadratic_loss",
    stoichiometry=Stoichiometry(reactants={"A": 1}, products={}),
    parameter_dependencies=("k",),
    implementation=lambda context: context.parameter("k") * context.state("A")**2,
)
case = Case(
    states={"A": A},
    parameters={"k": 2},
    reactions=(quadratic_loss,),
)
contribution = case.evaluate_reaction("quadratic_loss").contribution("A")

# Negative even immediately below A=0: this term pushes A farther negative.
assert contribution.subs({A: -0.001}) < 0
```

That observation is not yet classified as a failure or wrapped in a proof.
Those are later analysis layers consuming `EvaluatedReaction` objects.

## Deliberate scope

The foundation validates ids, sided stoichiometry, dependency availability,
hook outputs, and unique reaction selection. It does not impose mass-action
kinetics, aggregate component contributions, or select any solver
representation.

Potential next layers can therefore remain separate:

```text
user one-way reaction implementations
        -> Case / ReactionContext
        -> EvaluatedReaction objects
        -> sign/domain analysis
        -> diagnostics and proof records
```

## Development

```shell
PYTHONPATH=src .venv/bin/python -m unittest discover -v
```
