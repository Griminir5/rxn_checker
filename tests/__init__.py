"""Tests for rxn-checker."""

from rxn_checker import StateVariables, VariableBounds


def make_state_bounds(states: StateVariables) -> dict:
    bounds = {
        states.temperature: VariableBounds(200.0, 1500.0),
        states.pressure: VariableBounds(10_000.0, 10_000_000.0),
    }
    bounds.update(
        {
            concentration: VariableBounds(0.0, 1000.0, -0.1)
            for concentration in states.concentrations.values()
        }
    )
    return bounds
