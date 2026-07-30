"""Constructors for symbols with reaction-model semantics."""

from __future__ import annotations

from typing import Any

from sympy import Symbol


def _symbol_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string")
    name = name.strip()
    if not name:
        raise ValueError(f"{kind} name must not be empty")
    return name


def state(name: str) -> Symbol:
    """Create a real-valued concentration state.

    Concentrations are deliberately *not* declared non-negative.  The checker
    needs to reason about expressions just outside the physical domain, where a
    numerical solver may have produced a small negative value.
    """

    return Symbol(_symbol_name(name, "state"), real=True)


def parameter(name: str, **assumptions: Any) -> Symbol:
    """Create a real-valued model parameter.

    SymPy assumptions may be supplied when they are structural, for example
    ``parameter("k", positive=True)``.
    """

    if assumptions.get("real") is False:
        raise ValueError("reaction parameters must be real")
    assumptions["real"] = True
    return Symbol(_symbol_name(name, "parameter"), **assumptions)
