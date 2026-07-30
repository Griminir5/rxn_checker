"""Import-safe core objects for hookable symbolic reaction implementations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TypeAlias

from sympy import Expr, Symbol, sympify


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must not be blank or padded")
    return value


def _unique_identifiers(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label}s must be a sequence of strings, not one string")
    result = tuple(_identifier(value, label) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label}s must not contain duplicates")
    return result


def _expression(value: object, label: str) -> Expr:
    try:
        expression = sympify(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a SymPy expression") from exc
    if not isinstance(expression, Expr):
        raise TypeError(f"{label} must be a scalar SymPy expression")
    return expression


def _expression_mapping(
    values: Mapping[str, object], label: str
) -> Mapping[str, Expr]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, Expr] = {}
    for raw_id, raw_value in values.items():
        value_id = _identifier(raw_id, f"{label} id")
        result[value_id] = _expression(raw_value, f"{label}[{value_id!r}]")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ReactionContext:
    """The state expressions and parameter values visible to a reaction hook.

    Reaction implementations should use :meth:`state` and :meth:`parameter`
    rather than know anything about the enclosing model or solver.
    """

    states: Mapping[str, Expr]
    parameters: Mapping[str, Expr]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", _expression_mapping(self.states, "states"))
        object.__setattr__(
            self,
            "parameters",
            _expression_mapping(self.parameters, "parameters"),
        )

    def state(self, state_id: str) -> Expr:
        """Return the symbolic expression registered for a state id."""

        try:
            return self.states[state_id]
        except KeyError as exc:
            raise KeyError(f"state {state_id!r} is not available to this reaction") from exc

    def parameter(self, parameter_id: str) -> Expr:
        """Return the value or expression registered for a parameter id."""

        try:
            return self.parameters[parameter_id]
        except KeyError as exc:
            raise KeyError(
                f"parameter {parameter_id!r} is not available to this reaction"
            ) from exc


ExpressionHook: TypeAlias = Callable[[ReactionContext], object]


@dataclass(frozen=True, slots=True)
class Reaction:
    """A complete hookable reaction implementation.

    ``stoichiometry`` is sparse and uses state ids rather than solver objects.
    ``state_dependencies`` lists modifiers used by the expression but unchanged
    by the reaction; stoichiometric states are dependencies automatically.
    """

    id: str
    stoichiometry: Mapping[str, Expr]
    implementation: ExpressionHook
    state_dependencies: Sequence[str] = ()
    parameter_dependencies: Sequence[str] = ()
    name: str = ""
    source_reference: str = ""

    def __post_init__(self) -> None:
        reaction_id = _identifier(self.id, "reaction id")
        if not callable(self.implementation):
            raise TypeError(f"reaction {reaction_id!r} implementation must be callable")
        if not isinstance(self.stoichiometry, Mapping):
            raise TypeError("stoichiometry must be a mapping from state ids to coefficients")
        if not self.stoichiometry:
            raise ValueError(f"reaction {reaction_id!r} must define stoichiometry")

        stoichiometry: dict[str, Expr] = {}
        for raw_state_id, raw_coefficient in self.stoichiometry.items():
            state_id = _identifier(raw_state_id, "stoichiometric state id")
            coefficient = _expression(
                raw_coefficient,
                f"stoichiometric coefficient for {state_id!r}",
            )
            if (
                coefficient.free_symbols
                or coefficient.is_number is not True
                or coefficient.is_real is not True
                or coefficient.is_finite is not True
            ):
                raise ValueError(
                    f"stoichiometric coefficient for {state_id!r} must be a "
                    "finite real number"
                )
            if coefficient == 0:
                raise ValueError(
                    f"stoichiometric coefficient for {state_id!r} must be non-zero"
                )
            stoichiometry[state_id] = coefficient

        state_dependencies = _unique_identifiers(
            self.state_dependencies,
            "state dependency",
        )
        parameter_dependencies = _unique_identifiers(
            self.parameter_dependencies,
            "parameter dependency",
        )
        name = self.name or reaction_id
        _identifier(name, "reaction name")
        if self.source_reference:
            _identifier(self.source_reference, "source reference")

        object.__setattr__(self, "id", reaction_id)
        object.__setattr__(self, "stoichiometry", MappingProxyType(stoichiometry))
        object.__setattr__(self, "state_dependencies", state_dependencies)
        object.__setattr__(self, "parameter_dependencies", parameter_dependencies)
        object.__setattr__(self, "name", name)

    @property
    def required_state_ids(self) -> tuple[str, ...]:
        """Stoichiometric states followed by non-stoichiometric dependencies."""

        return tuple(dict.fromkeys((*self.stoichiometry, *self.state_dependencies)))

    @property
    def required_parameter_ids(self) -> tuple[str, ...]:
        return tuple(self.parameter_dependencies)

    def evaluate(self, context: ReactionContext) -> EvaluatedReaction:
        """Invoke the hook with only its declared dependencies visible."""

        missing_states = set(self.required_state_ids) - set(context.states)
        missing_parameters = set(self.required_parameter_ids) - set(context.parameters)
        if missing_states or missing_parameters:
            differences: list[str] = []
            if missing_states:
                differences.append(
                    "missing states "
                    + ", ".join(sorted(missing_states))
                )
            if missing_parameters:
                differences.append(
                    "missing parameters "
                    + ", ".join(sorted(missing_parameters))
                )
            raise ValueError(
                f"reaction {self.id!r} cannot be evaluated: {'; '.join(differences)}"
            )

        scoped_context = ReactionContext(
            states={
                state_id: context.states[state_id]
                for state_id in self.required_state_ids
            },
            parameters={
                parameter_id: context.parameters[parameter_id]
                for parameter_id in self.required_parameter_ids
            },
        )
        expression = _expression(
            self.implementation(scoped_context),
            f"expression returned by reaction {self.id!r}",
        )

        allowed_symbols: set[Symbol] = set()
        for value in (
            *scoped_context.states.values(),
            *scoped_context.parameters.values(),
        ):
            allowed_symbols.update(value.free_symbols)
        undeclared_symbols = expression.free_symbols - allowed_symbols
        if undeclared_symbols:
            unknown = ", ".join(sorted(str(symbol) for symbol in undeclared_symbols))
            raise ValueError(
                f"reaction {self.id!r} expression contains symbols outside its "
                f"declared dependencies: {unknown}"
            )
        return EvaluatedReaction(reaction=self, expression=expression)


@dataclass(frozen=True, slots=True)
class EvaluatedReaction:
    """A reaction expression retaining its stoichiometry and provenance."""

    reaction: Reaction
    expression: Expr

    @property
    def id(self) -> str:
        return self.reaction.id

    @property
    def stoichiometry(self) -> Mapping[str, Expr]:
        return self.reaction.stoichiometry

    def contribution(self, state_id: str) -> Expr:
        """Return ``coefficient * expression`` for one state id.

        This is an atomic contribution only. No equation system, ODE, or other
        aggregate mathematical object is implied.
        """

        return self.stoichiometry.get(state_id, sympify(0)) * self.expression


@dataclass(frozen=True, slots=True)
class Case:
    """A neutral container into which reaction implementations are hooked.

    State entries are symbolic expressions. Parameter entries may be concrete
    values or symbolic expressions. The case evaluates reactions independently
    and does not assemble an ODE or choose a solver representation.
    """

    states: Mapping[str, Expr]
    parameters: Mapping[str, Expr]
    reactions: Sequence[Reaction]
    name: str = "case"

    def __post_init__(self) -> None:
        _identifier(self.name, "case name")
        states = _expression_mapping(self.states, "states")
        if not states:
            raise ValueError("a case must declare at least one state")
        parameters = _expression_mapping(self.parameters, "parameters")
        overlap = set(states) & set(parameters)
        if overlap:
            raise ValueError(
                "state and parameter ids must be disjoint: "
                + ", ".join(sorted(overlap))
            )

        reactions = tuple(self.reactions)
        if any(not isinstance(reaction, Reaction) for reaction in reactions):
            raise TypeError("all case reactions must be Reaction objects")
        reaction_ids = [reaction.id for reaction in reactions]
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("reaction ids must be unique within a case")

        for reaction in reactions:
            missing_states = set(reaction.required_state_ids) - set(states)
            missing_parameters = set(reaction.required_parameter_ids) - set(parameters)
            if missing_states or missing_parameters:
                differences: list[str] = []
                if missing_states:
                    differences.append(
                        "missing states " + ", ".join(sorted(missing_states))
                    )
                if missing_parameters:
                    differences.append(
                        "missing parameters " + ", ".join(sorted(missing_parameters))
                    )
                raise ValueError(
                    f"reaction {reaction.id!r} is incompatible with case "
                    f"{self.name!r}: {'; '.join(differences)}"
                )

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "reactions", reactions)

    @property
    def context(self) -> ReactionContext:
        return ReactionContext(states=self.states, parameters=self.parameters)

    def reaction(self, reaction_id: str) -> Reaction:
        """Look up a hooked reaction by id."""

        for reaction in self.reactions:
            if reaction.id == reaction_id:
                return reaction
        raise KeyError(reaction_id)

    def evaluate_reaction(self, reaction_id: str) -> EvaluatedReaction:
        """Evaluate one hooked reaction by id."""

        return self.reaction(reaction_id).evaluate(self.context)

    def evaluate_reactions(self) -> tuple[EvaluatedReaction, ...]:
        """Evaluate all hooks in deterministic case order."""

        context = self.context
        return tuple(reaction.evaluate(context) for reaction in self.reactions)

    @property
    def expressions(self) -> Mapping[str, Expr]:
        """Evaluated expressions keyed by reaction id."""

        return MappingProxyType(
            {evaluated.id: evaluated.expression for evaluated in self.evaluate_reactions()}
        )


__all__ = (
    "Case",
    "EvaluatedReaction",
    "ExpressionHook",
    "Reaction",
    "ReactionContext",
)
