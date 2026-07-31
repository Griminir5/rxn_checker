# rxn-checker

`rxn-checker` checks symbolic reaction implementations for physical and
numerical robustness.

The current foundation has three small pieces:

- `rxn_checker.species` stores species properties.
- `rxn_checker` loads a case and owns all symbolic state variables.
- `rxn_checker.reactions` defines one-way reactions using those state variables.

## Case-owned state variables

A case lists its species and reaction selectors:

```yaml
species:
  - Aye
  - Bee
reactions:
  - aye_to_bee.simple
```

A selector has one of two forms:

- `family.reaction` selects one reaction from a family;
- `family` selects every reaction in that family.

For example, replacing `aye_to_bee.simple` above with `aye_to_bee` selects
both `aye_to_bee.simple` and `aye_to_bee.autocatalytic`.

`load_case()` creates one `StateVariables` object containing the concentration
symbol for every listed species, plus temperature and pressure:

```python
from rxn_checker import load_case

case = load_case("example_case/case.yaml")

aye = case.states.concentration("Aye")
bee = case.states.concentration("Bee")
temperature = case.states.temperature
pressure = case.states.pressure
```

Concentration symbols are real but are not assumed non-negative, because later
checks need to inspect expressions just outside the physical domain.

## Reaction family modules

The family part of a selector maps directly to a same-named packaged module.
For example, both `aye_to_bee` and `aye_to_bee.simple` load
`rxn_checker/reactions/aye_to_bee.py`. Each family module exposes a `REACTIONS`
mapping from its local reaction names to builder functions:

```python
from types import MappingProxyType

from rxn_checker import Reaction

RATE_CONSTANT = 2.0


def build_simple(states):
    aye = states.concentration("Aye")

    return Reaction(
        id="aye_to_bee.simple",
        family="aye_to_bee",
        reactants={"Aye": 1},
        products={"Bee": 1},
        rate=RATE_CONSTANT * aye,
    )


REACTIONS = MappingProxyType({"simple": build_simple})
```

The package automatically discovers these family modules and combines their
local mappings into a unified builder registry. There is no central list to
maintain: adding a family file is enough to register it. Discovery imports the
family modules and records their builder functions, but it does not build any
`Reaction` objects. A qualified selector calls one builder, while a bare family
selector calls every builder in that family's mapping order.

The case passes its state object into each selected builder, so reaction code
does not create SymPy symbols. Constants and correlations stay as ordinary
numbers and Python code inside the reaction module. Consequently, every free
SymPy symbol in a rate represents a state variable.

After building a reaction, the loader verifies its fully qualified id and
family. `Case` then verifies that its species exist and every free rate symbol
belongs to the case. Overlapping selectors such as `aye_to_bee` together with
`aye_to_bee.simple` are rejected rather than loading a reaction twice.

## Reaction definition

A `Reaction` represents one forward direction and contains:

- a unique implementation id;
- a family id for implementations of the same net reaction;
- separate reactant and product mappings;
- optional non-consumed catalysts;
- the resulting SymPy rate expression.

Net stoichiometry is derived. The authoritative sided mappings preserve the
difference between these mechanisms:

```text
Aye + Bee -> 2 Bee
Aye       -> Bee
```

Both have net `Aye: -1, Bee: +1`, but their rate expressions and boundary
behaviour are different. A reversible expression is not represented by one
reaction; each direction must be a separate `Reaction`.

Species existence, atom balance, family comparison, and physical/numerical
properties are later checks. They are intentionally not responsibilities of
the species registry or reaction-definition object.

## Development

```shell
PYTHONPATH=src .venv/bin/python -m unittest discover -v
```
