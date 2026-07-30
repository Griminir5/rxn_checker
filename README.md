# rxn-checker

`rxn-checker` supplies the import-safe objects underneath checks for whether
reaction expressions and reaction networks are friendly to numerical solvers.
For example, a later checker can ask whether a reaction pushes a tiny negative
component concentration back toward the physical region or farther away from
it.

This package currently defines the integration seam. It does not assume that
the consumer is building an ODE: it evaluates reaction implementations one at a
time and retains their expressions, stoichiometry, and provenance.

## The four building blocks

- `ReactionContext` is the narrow interface through which an implementation
  accesses registered states and parameters.
- `Reaction` holds an id, sparse net stoichiometry, declared dependencies, and
  a callable implementation.
- `Case` registers the state expressions and parameter values, selects
  reactions, validates their requirements, and invokes their hooks.
- `EvaluatedReaction` is the neutral output: one expression tied to its
  reaction metadata. It can report a stoichiometry-weighted contribution for a
  component, but it does not assemble an equation system.

This follows the same boundary used by `multisolid-CL`: metadata remains safe to
inspect, implementations receive a context instead of importing the enclosing
model, and the general object resolves the hooks.

## Defining and hooking in a reaction

```python
from rxn_checker import Case, Reaction, ReactionContext, parameter, state


def quadratic_loss(context: ReactionContext):
    """A complete symbolic implementation supplied by a user."""
    A = context.state("A")
    k = context.parameter("k")
    return k * A**2


loss = Reaction(
    id="quadratic_loss",
    name="Quadratic loss of A",
    stoichiometry={"A": -1, "B": 1},  # products minus reactants
    parameter_dependencies=("k",),
    implementation=quadratic_loss,
    source_reference="example",
)

A = state("A")
B = state("B")
k = parameter("k", positive=True)

symbolic_case = Case(
    name="symbolic",
    states={"A": A, "B": B},
    parameters={"k": k},
    reactions=(loss,),
)

evaluated = symbolic_case.evaluate_reaction("quadratic_loss")
assert evaluated.expression == k * A**2
assert evaluated.contribution("A") == -k * A**2
assert evaluated.contribution("B") == k * A**2
```

Parameter entries are values, not necessarily symbols, so the same reaction can
be attached to a concrete case without changing its implementation:

```python
concrete_case = Case(
    states={"A": A, "B": B},
    parameters={"k": 2.0},
    reactions=(loss,),
)

assert concrete_case.expressions["quadratic_loss"] == 2.0 * A**2
```

Callable objects work as implementations too. This lets a larger user-owned
reaction object carry correlations or supporting methods while presenting the
same single `__call__(context)` hook.

## Declaring dependencies

Every state named in stoichiometry is automatically visible to the hook.
Additional state dependencies—catalysts, inhibitors, temperatures, or other
unchanged quantities—must be listed explicitly:

```python
def catalysed(context: ReactionContext):
    return (
        context.parameter("k")
        * context.state("A")
        * context.state("catalyst")
    )


catalysed_loss = Reaction(
    id="catalysed_loss",
    stoichiometry={"A": -1},
    state_dependencies=("catalyst",),
    parameter_dependencies=("k",),
    implementation=catalysed,
)
```

A hook receives only its declared dependencies. Missing case entries are
reported when the case is created; attempts to access undeclared entries are
reported when the hook runs. The returned expression is also checked for
closed-over symbols outside the declared dependency set.

## Why states are not assumed non-negative

`state("A")` creates a real SymPy symbol but deliberately does not give it a
non-negative assumption. A non-negative assumption could simplify away exactly
the behaviour the checker needs to inspect.

```python
case = Case(
    states={"A": A, "B": B},
    parameters={"k": 2},
    reactions=(loss,),
)
contribution = case.evaluate_reaction("quadratic_loss").contribution("A")

# Negative even immediately below A=0: this term pushes A farther negative.
assert contribution.subs({A: -0.001}) < 0
```

That observation is not yet classified as a failure or wrapped in a proof.
Those are later analysis layers consuming `EvaluatedReaction` objects.

## Deliberate scope

The foundation validates ids, dependency availability, hook outputs, unique
reaction selection, and finite real constant stoichiometry. It does not impose
mass-action kinetics, require a reaction expression to be positive, aggregate
component contributions, or select any solver representation.

Potential next layers can therefore remain separate:

```text
user reaction implementations
        -> Case / ReactionContext
        -> EvaluatedReaction objects
        -> sign/domain analysis
        -> diagnostics and proof records
```

## Development

```shell
PYTHONPATH=src .venv/bin/python -m unittest discover -v
```
