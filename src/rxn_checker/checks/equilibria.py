"""Exact symbolic relationships for all physical kinetic steady states.

The check describes the equilibrium set instead of asking SymPy for explicit
roots.  Its result is read in this order: choose a branch, define that branch's
helper symbols, set its balance equations to zero, and apply its conditions.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import sympy as sp

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
)


# Public result -------------------------------------------------------------


@dataclass(frozen=True)
class EquilibriumHelper:
    """A named expression and the exact constraints that define it."""

    symbol: sp.Symbol
    expression: sp.Expr
    kind: str
    equation: sp.Expr
    nonzero: tuple[sp.Expr, ...] = ()
    conditions: tuple[sp.Expr, ...] = ()


@dataclass(frozen=True)
class EquilibriumBranch:
    """One alternative in the complete equilibrium relationship."""

    label: str
    helpers: tuple[EquilibriumHelper, ...]
    balances: tuple[sp.Expr, ...]
    nonzero: tuple[sp.Expr, ...]
    conditions: tuple[sp.Expr, ...]

    @property
    def equations(self) -> tuple[sp.Expr, ...]:
        """All equations, ordered as helper definitions then balances."""

        definitions = tuple(helper.equation for helper in self.helpers)
        return definitions + self.balances


@dataclass(frozen=True)
class EquilibriumRelation:
    """Complete physical steady-state set represented as exact branches.

    The represented set is ``physical_domain AND OR(branches)``. Helper
    symbols are existential coordinates local to their branch.
    """

    species: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    source_species: tuple[str, ...]
    physical_domain: tuple[sp.Expr, ...]
    branches: tuple[EquilibriumBranch, ...]
    algebraic: bool
    source_equations: int
    independent_balances: int
    source_operations: int
    relation_operations: int
    diagnostic: str | None = None

    @property
    def helpers(self) -> tuple[EquilibriumHelper, ...]:
        """All distinct helpers, in first-use order."""

        found: dict[sp.Symbol, EquilibriumHelper] = {}
        for branch in self.branches:
            for helper in branch.helpers:
                found.setdefault(helper.symbol, helper)
        return tuple(found.values())


# Exact source reduction ----------------------------------------------------


def _rational(value: object) -> sp.Rational:
    return sp.Rational(str(value))


def _network(case: Case) -> tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]:
    matrix = sp.ImmutableMatrix(
        [
            [
                _rational(reaction.net_stoichiometry.get(species_id, 0))
                for reaction in case.reactions
            ]
            for species_id in case.states.species_ids
        ]
    )
    rates = sp.ImmutableMatrix([reaction.rate for reaction in case.reactions])
    return matrix, sp.ImmutableMatrix(matrix * rates)


def _independent_rows(
    matrix: sp.MatrixBase,
    source: sp.MatrixBase,
) -> tuple[int, ...]:
    """Choose a low-complexity basis of stoichiometric source rows."""

    target_rank = int(matrix.rank())
    selected: list[int] = []
    rank = 0
    rows = sorted(
        range(matrix.rows),
        key=lambda row: (int(sp.count_ops(source[row])), row),
    )
    for row in rows:
        trial = matrix.extract((*selected, row), range(matrix.cols))
        trial_rank = int(trial.rank())
        if trial_rank > rank:
            selected.append(row)
            rank = trial_rank
        if rank == target_rank:
            break
    return tuple(selected)


def _physical_domain(case: Case) -> tuple[sp.Expr, ...]:
    conditions = []
    for symbol, bounds in case.state_bounds.items():
        conditions.extend(
            (
                sp.Ge(symbol, _rational(bounds.physical_lower)),
                sp.Le(symbol, _rational(bounds.physical_upper)),
            )
        )
    return tuple(conditions)


def _physical_symbols(case: Case) -> Mapping[sp.Symbol, sp.Symbol]:
    """Mirror configured bounds as assumptions used only for safe rewrites."""

    replacements = {}
    for symbol, bounds in case.state_bounds.items():
        if bounds.physical_lower > 0:
            assumptions = {"positive": True}
        elif bounds.physical_lower == 0:
            assumptions = {"nonnegative": True}
        elif bounds.physical_upper < 0:
            assumptions = {"negative": True}
        elif bounds.physical_upper == 0:
            assumptions = {"nonpositive": True}
        else:
            assumptions = {"real": True}
        replacements[symbol] = sp.Dummy(symbol.name, **assumptions)
    return replacements


def _with_physical_assumptions(
    expression: sp.Expr,
    replacements: Mapping[sp.Symbol, sp.Symbol],
) -> sp.Expr:
    return expression.xreplace(
        {
            symbol: replacement
            for symbol, replacement in replacements.items()
            if expression.has(symbol)
        }
    )


def _proved(
    expression: sp.Expr,
    replacements: Mapping[sp.Symbol, sp.Symbol],
    property_name: str,
) -> bool:
    assumed = _with_physical_assumptions(expression, replacements)
    return getattr(assumed, f"is_{property_name}") is True


def _rewrite_physical_clamps(expression: sp.Expr, case: Case) -> sp.Expr:
    """Use clamp identities only when configured bounds prove them."""

    replacements: dict[sp.Expr, sp.Expr] = {}
    for function in expression.atoms(sp.Max, sp.Min):
        arguments = tuple(argument for argument in function.args if argument != 0)
        if len(arguments) != 1 or 0 not in function.args:
            continue
        argument = arguments[0]
        if not isinstance(argument, sp.Symbol) or argument not in case.state_bounds:
            continue
        bounds = case.state_bounds[argument]
        if isinstance(function, sp.Max) and bounds.physical_lower >= 0:
            replacements[function] = argument
        if isinstance(function, sp.Min) and bounds.physical_upper <= 0:
            replacements[function] = argument
    return expression.xreplace(replacements)


# Algebraic expression lifting ---------------------------------------------


def _small_rational(expression: sp.Expr) -> sp.Rational | None:
    """Recognize exact small rational exponents without guessing decimals."""

    if isinstance(expression, sp.Rational):
        return expression
    if isinstance(expression, sp.Float):
        exact = sp.Rational(expression)
        if exact.q <= 64:
            return exact
    return None


class _ExpressionCompiler:
    """Turn supported state expressions into exact helper graphs."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self.concentrations = frozenset(case.states.concentrations.values())
        self.parameters = frozenset((case.states.temperature, case.states.pressure))
        self.assumptions = _physical_symbols(case)
        self.helpers: list[EquilibriumHelper] = []
        self.denominators: list[sp.Expr] = []
        self.state_helpers: set[sp.Symbol] = set()
        self.roots: dict[tuple[sp.Expr, int], sp.Symbol] = {}
        self.functions: dict[sp.Expr, sp.Symbol] = {}
        self.coefficients: dict[sp.Expr, sp.Symbol] = {}
        self.counts: Counter[str] = Counter()

    def _symbol(self, kind: str, **assumptions: bool) -> sp.Symbol:
        self.counts[kind] += 1
        return sp.Dummy(f"{kind}_{self.counts[kind]}", **assumptions)

    def _depends_on_concentration(self, expression: sp.Expr) -> bool:
        state_symbols = self.concentrations | self.state_helpers
        return bool(expression.free_symbols & state_symbols)

    def _numerator(self, expression: sp.Expr) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
        numerator, denominator = sp.together(expression).as_numer_denom()
        nonzero = ()
        if not _proved(denominator, self.assumptions, "nonzero"):
            nonzero = (denominator,)
        return numerator, nonzero

    def clear_rate_denominator(self, expression: sp.Expr) -> sp.Expr:
        numerator, nonzero = self._numerator(expression)
        self.denominators.extend(nonzero)
        return numerator

    def _add_helper(
        self,
        symbol: sp.Symbol,
        expression: sp.Expr,
        kind: str,
        equation: sp.Expr,
        *,
        nonzero: Sequence[sp.Expr] = (),
        conditions: Sequence[sp.Expr] = (),
        state_dependent: bool = False,
    ) -> sp.Symbol:
        if state_dependent:
            self.state_helpers.add(symbol)
        numerator, denominator_nonzero = self._numerator(equation)
        self.helpers.append(
            EquilibriumHelper(
                symbol=symbol,
                expression=expression,
                kind=kind,
                equation=numerator,
                nonzero=_unique(tuple(nonzero) + denominator_nonzero),
                conditions=tuple(conditions),
            )
        )
        return symbol

    def _root(self, base: sp.Expr, exponent: sp.Rational) -> sp.Expr:
        degree = int(exponent.q)
        key = (base, degree)
        root = self.roots.get(key)
        if root is None:
            positive = _proved(base, self.assumptions, "positive")
            root = self._symbol(
                "root",
                **({"positive": True} if positive else {"nonnegative": True}),
            )
            self.roots[key] = root
            conditions = [
                (sp.Gt if positive else sp.Ge)(root, 0, evaluate=False)
            ]
            if not _proved(base, self.assumptions, "nonnegative"):
                conditions.append(sp.Ge(base, 0))
            self._add_helper(
                root,
                base,
                "root",
                root**degree - base,
                conditions=conditions,
                state_dependent=True,
            )
        if exponent.p < 0:
            # The branch builder retains this condition whenever the root is used.
            helper_index = next(
                index
                for index, helper in enumerate(self.helpers)
                if helper.symbol == root
            )
            helper = self.helpers[helper_index]
            self.helpers[helper_index] = replace(
                helper,
                nonzero=_unique(helper.nonzero + (root,)),
            )
        return root ** int(exponent.p)

    def _ordered_value(
        self,
        expression: sp.Expr,
        arguments: tuple[sp.Expr, ...],
    ) -> sp.Expr:
        if isinstance(expression, sp.Max):
            kind, relation = "maximum", sp.Ge
        else:
            kind, relation = "minimum", sp.Le
        value = self._symbol(kind, real=True)
        return self._add_helper(
            value,
            expression,
            kind,
            sp.prod(value - argument for argument in arguments),
            conditions=tuple(relation(value, argument) for argument in arguments),
            state_dependent=True,
        )

    def _absolute_value(self, expression: sp.Expr, argument: sp.Expr) -> sp.Expr:
        value = self._symbol("absolute", nonnegative=True)
        return self._add_helper(
            value,
            expression,
            "absolute value",
            value**2 - argument**2,
            conditions=(sp.Ge(value, 0, evaluate=False),),
            state_dependent=True,
        )

    def _function_value(self, expression: sp.Expr) -> sp.Expr:
        value = self.functions.get(expression)
        if value is not None:
            return value
        value = self._symbol(
            "function",
            **({"positive": True} if expression.func is sp.exp else {"real": True}),
        )
        self.functions[expression] = value
        return self._add_helper(
            value,
            expression,
            "state function",
            value - expression,
            state_dependent=True,
        )

    def _coefficient(self, expression: sp.Expr) -> sp.Expr:
        value = self.coefficients.get(expression)
        if value is not None:
            return value
        value = self._symbol(
            "coefficient",
            **({"positive": True} if expression.func is sp.exp else {"real": True}),
        )
        self.coefficients[expression] = value
        return self._add_helper(
            value,
            expression,
            "parameter coefficient",
            value - expression,
        )

    def lift(self, expression: sp.Expr) -> sp.Expr:
        if not expression.args:
            return expression
        arguments = tuple(self.lift(argument) for argument in expression.args)
        rebuilt = expression.func(*arguments)

        if isinstance(rebuilt, sp.Pow):
            exponent = _small_rational(rebuilt.exp)
            if (
                exponent is not None
                and exponent.q != 1
                and self._depends_on_concentration(rebuilt.base)
            ):
                return self._root(rebuilt.base, exponent)
            if (
                self._depends_on_concentration(rebuilt)
                and rebuilt.exp.is_integer is not True
            ):
                return self._function_value(rebuilt)
            return rebuilt
        if isinstance(rebuilt, (sp.Max, sp.Min)) and self._depends_on_concentration(
            rebuilt
        ):
            return self._ordered_value(rebuilt, arguments)
        if isinstance(rebuilt, sp.Abs) and self._depends_on_concentration(rebuilt):
            return self._absolute_value(rebuilt, arguments[0])
        if rebuilt.is_Function or isinstance(rebuilt, sp.Piecewise):
            if self._depends_on_concentration(rebuilt):
                return self._function_value(rebuilt)
            if rebuilt.free_symbols & self.parameters:
                return self._coefficient(rebuilt)
        return rebuilt

    def required_helpers(
        self,
        expressions: Sequence[sp.Expr],
    ) -> tuple[EquilibriumHelper, ...]:
        """Return the transitive helper definitions used by a branch."""

        live = set().union(*(expression.free_symbols for expression in expressions))
        changed = True
        while changed:
            changed = False
            for helper in self.helpers:
                if helper.symbol not in live:
                    continue
                dependencies = helper.expression.free_symbols - live
                if dependencies:
                    live.update(dependencies)
                    changed = True
        return tuple(helper for helper in self.helpers if helper.symbol in live)


# Exact factor branches -----------------------------------------------------


def _factors(
    expression: sp.Expr,
    assumptions: Mapping[sp.Symbol, sp.Symbol],
) -> list[sp.Expr]:
    return [
        factor
        for factor in sp.Mul.make_args(sp.factor_terms(expression))
        if not _proved(factor, assumptions, "nonzero")
    ]


def _zero_factor(expression: sp.Expr) -> sp.Expr:
    if (
        isinstance(expression, sp.Pow)
        and expression.exp.is_integer is True
        and expression.exp.is_positive is True
    ):
        return expression.base
    return expression


def _common_factors(factor_lists: Sequence[Sequence[sp.Expr]]) -> tuple[sp.Expr, ...]:
    if not factor_lists:
        return ()
    normalized = [tuple(map(_zero_factor, factors)) for factors in factor_lists]
    return tuple(
        dict.fromkeys(
            factor
            for factor in normalized[0]
            if all(factor in others for others in normalized[1:])
        )
    )


def _remove_factors(factors: Sequence[sp.Expr], common: Sequence[sp.Expr]) -> sp.Expr:
    remaining = list(factors)
    for wanted in common:
        match = next(factor for factor in remaining if _zero_factor(factor) == wanted)
        remaining.remove(match)
    return sp.Mul(*remaining)


def _unique(expressions: Sequence[sp.Expr]) -> tuple[sp.Expr, ...]:
    return tuple(dict.fromkeys(expressions))


def _make_branch(
    compiler: _ExpressionCompiler,
    label: str,
    balances: Sequence[sp.Expr],
    nonzero: Sequence[sp.Expr],
) -> EquilibriumBranch:
    seeds = (
        tuple(balances)
        + tuple(nonzero)
        + tuple(compiler.denominators)
    )
    helpers = compiler.required_helpers(seeds)
    helper_nonzero = tuple(
        condition for helper in helpers for condition in helper.nonzero
    )
    helper_conditions = tuple(
        condition for helper in helpers for condition in helper.conditions
    )
    return EquilibriumBranch(
        label=label,
        helpers=helpers,
        balances=tuple(balances),
        nonzero=_unique(tuple(compiler.denominators) + helper_nonzero + tuple(nonzero)),
        conditions=_unique(helper_conditions),
    )


# Public construction -------------------------------------------------------


def check_equilibria(case: Case) -> EquilibriumRelation:
    """Build an exact, domain-aware relationship for every steady state."""

    matrix, source = _network(case)
    source_equations = tuple(expression for expression in source if expression != 0)
    selected_rows = _independent_rows(matrix, source)
    selected = tuple(
        (row, source[row]) for row in selected_rows if source[row] != 0
    )

    compiler = _ExpressionCompiler(case)
    numerators = []
    for _, expression in selected:
        lifted = compiler.lift(_rewrite_physical_clamps(expression, case))
        numerators.append(compiler.clear_rate_denominator(lifted))

    factor_lists = tuple(
        _factors(numerator, compiler.assumptions) for numerator in numerators
    )
    common = _common_factors(factor_lists)
    core = tuple(_remove_factors(factors, common) for factors in factor_lists)

    branches = [
        _make_branch(compiler, f"{sp.sstr(factor)} = 0", (factor,), ())
        for factor in common
    ]
    if not any(equation.is_number and equation != 0 for equation in core):
        label = "remaining shared factors nonzero" if common else "equilibrium balances"
        branches.append(_make_branch(compiler, label, core, common))

    species = tuple(case.states.concentrations.values())
    helper_symbols = tuple(
        dict.fromkeys(
            helper.symbol
            for branch in branches
            for helper in branch.helpers
        )
    )
    variables = species + helper_symbols
    algebraic = all(
        equation.is_polynomial(*variables)
        for branch in branches
        for equation in branch.equations
    )
    diagnostic = None if algebraic else (
        "Concentration-dependent nonalgebraic functions remain as exact definitions."
    )
    relation_operations = sum(
        int(sp.count_ops(expression))
        for branch in branches
        for expression in branch.equations + branch.nonzero + branch.conditions
    )
    return EquilibriumRelation(
        species=species,
        parameters=(case.states.temperature, case.states.pressure),
        source_species=tuple(case.states.species_ids[row] for row, _ in selected),
        physical_domain=_physical_domain(case),
        branches=tuple(branches),
        algebraic=algebraic,
        source_equations=len(source_equations),
        independent_balances=len(selected),
        source_operations=sum(
            int(sp.count_ops(expression)) for expression in source_equations
        ),
        relation_operations=relation_operations,
        diagnostic=diagnostic,
    )


# Report rendering ----------------------------------------------------------


_REPORT_WIDTH = 100


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or singular + "s")
    return f"{count} {word}"


def _helper_aliases(
    helpers: Sequence[EquilibriumHelper],
) -> dict[sp.Symbol, sp.Symbol]:
    """Give internal helper symbols short names for the text report."""

    counts: Counter[str] = Counter()
    aliases: dict[sp.Symbol, sp.Symbol] = {}
    for helper in helpers:
        group = {
            "root": "root",
            "parameter coefficient": "k",
            "state function": "f",
            "absolute value": "abs",
        }.get(helper.kind, helper.kind)
        counts[group] += 1
        number = counts[group]
        if group == "root" and number <= 3:
            name = ("u", "v", "w")[number - 1]
        else:
            name = f"{group}{number}"
        aliases[helper.symbol] = sp.Symbol(name)
    return aliases


def _helper_value(helper: EquilibriumHelper) -> sp.Expr:
    if helper.kind != "root":
        return helper.expression
    degree = int(sp.degree(helper.equation, helper.symbol))
    return helper.expression ** sp.Rational(1, degree)


def _expand_replacements(
    expression: sp.Expr,
    replacements: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr:
    while expression.free_symbols.intersection(replacements):
        updated = expression.xreplace(replacements)
        if updated == expression:
            break
        expression = updated
    return expression


def _compact_expressions(
    expressions: Sequence[sp.Expr],
) -> tuple[list[tuple[sp.Symbol, sp.Expr]], list[sp.Expr]]:
    """Name repeated additive expressions and inline trivial CSE temporaries."""

    replacements, reduced = sp.cse(
        expressions,
        symbols=sp.numbered_symbols("_shared_"),
        order="canonical",
    )
    discarded: dict[sp.Symbol, sp.Expr] = {}
    visible: dict[sp.Symbol, sp.Symbol] = {}
    definitions: list[tuple[sp.Symbol, sp.Expr]] = []
    occupied = {symbol.name for expression in expressions for symbol in expression.free_symbols}

    for temporary, raw_expression in replacements:
        expression = _expand_replacements(raw_expression, discarded)
        if isinstance(expression, sp.Add) and sp.count_ops(expression) >= 5:
            number = len(definitions) + 1
            name = f"s{number}"
            while name in occupied:
                number += 1
                name = f"s{number}"
            symbol = sp.Symbol(name)
            occupied.add(name)
            visible[temporary] = symbol
            definitions.append((symbol, expression.xreplace(visible)))
        else:
            discarded[temporary] = expression

    compact = [
        _expand_replacements(expression, discarded).xreplace(visible)
        for expression in reduced
    ]
    return definitions, compact


def _ordered_definitions(
    definitions: Sequence[tuple[sp.Symbol, sp.Expr]],
) -> list[tuple[sp.Symbol, sp.Expr]]:
    """Put each definition after the definitions on which it depends."""

    pending = list(definitions)
    symbols = {symbol for symbol, _ in pending}
    ordered: list[tuple[sp.Symbol, sp.Expr]] = []
    emitted: set[sp.Symbol] = set()
    while pending:
        for index, (symbol, expression) in enumerate(pending):
            if expression.free_symbols.intersection(symbols) <= emitted:
                ordered.append(pending.pop(index))
                emitted.add(symbol)
                break
        else:
            ordered.extend(pending)
            break
    return ordered


def _text(expression: sp.Expr) -> str:
    return sp.sstr(expression).replace("**", "^")


def _product_lines(expression: sp.Expr, prefix: str) -> list[str]:
    factors = sp.Mul.make_args(expression)
    rendered = [
        f"({_text(factor)})" if isinstance(factor, sp.Add) else _text(factor)
        for factor in factors
    ]
    lines: list[str] = []
    current = prefix
    for index, factor in enumerate(rendered):
        token = factor if index == 0 else f"*{factor}"
        if len(current) + len(token) > _REPORT_WIDTH and current != prefix:
            lines.append(current)
            current = " " * len(prefix) + f"*{factor}"
        else:
            current += token
    lines.append(current)
    return lines


def _expression_lines(expression: sp.Expr, indent: str) -> list[str]:
    terms = expression.as_ordered_terms() if isinstance(expression, sp.Add) else [expression]
    lines: list[str] = []
    for index, term in enumerate(terms):
        negative = term.could_extract_minus_sign()
        unsigned = -term if negative else term
        if negative:
            sign = "- "
        elif index:
            sign = "+ "
        else:
            sign = ""
        lines.extend(_product_lines(unsigned, indent + sign))
    return lines


def _definition_lines(symbol: sp.Symbol, expression: sp.Expr) -> list[str]:
    prefix = f"    {symbol} = "
    rendered = _text(expression)
    if len(prefix) + len(rendered) <= _REPORT_WIDTH:
        return [prefix + rendered]
    if expression.func is sp.exp and isinstance(expression.args[0], sp.Add):
        return (
            [prefix + "exp("]
            + _expression_lines(expression.args[0], "      ")
            + ["    )"]
        )
    return [prefix.rstrip()] + _expression_lines(expression, "      ")


def _nonzero_text(expression: sp.Expr, result: EquilibriumRelation) -> str:
    if isinstance(expression, sp.Symbol):
        for condition in result.physical_domain:
            if condition.lhs != expression:
                continue
            if isinstance(condition, sp.GreaterThan) and condition.rhs >= 0:
                return f"{_text(expression)} > 0"
            if isinstance(condition, sp.LessThan) and condition.rhs <= 0:
                return f"{_text(expression)} < 0"
    return f"{_text(expression)} != 0"


def _branch_details(
    branch: EquilibriumBranch,
    index: int,
    result: EquilibriumRelation,
) -> list[str]:
    aliases = _helper_aliases(branch.helpers)
    occupied = {symbol.name for symbol in result.species} | {
        symbol.name for symbol in aliases.values()
    }
    for parameter, short_name in zip(result.parameters, ("T", "p"), strict=False):
        if short_name not in occupied:
            aliases[parameter] = sp.Symbol(short_name)
            occupied.add(short_name)
    helper_symbols = {helper.symbol for helper in branch.helpers}
    if (
        len(branch.balances) == 1
        and not branch.helpers
        and not branch.nonzero
        and not branch.conditions
    ):
        equation = branch.balances[0].xreplace(aliases)
        return [f"Branch {index}: {_text(equation)} = 0."]
    if not branch.equations and not branch.nonzero and not branch.conditions:
        return [f"Branch {index}: the entire configured physical domain."]

    external_nonzero = tuple(
        expression
        for expression in branch.nonzero
        if not expression.free_symbols.intersection(helper_symbols)
    )
    heading = f"Branch {index}"
    heading_nonzero = None
    if len(external_nonzero) == 1:
        heading_nonzero = external_nonzero[0]
        heading += f": {_nonzero_text(heading_nonzero, result)}"
    lines = [heading + "."]

    helper_values = [
        _helper_value(helper).xreplace(aliases) for helper in branch.helpers
    ]
    root_constraints = [
        helper.equation.xreplace(aliases)
        for helper in branch.helpers
        if helper.kind == "root"
    ]
    balances = [balance.xreplace(aliases) for balance in branch.balances]
    shared, compact = _compact_expressions(
        helper_values + root_constraints + balances
    )
    value_count = len(helper_values)
    constraint_count = len(root_constraints)
    helper_values = compact[:value_count]
    root_constraints = compact[value_count : value_count + constraint_count]
    balances = compact[value_count + constraint_count :]

    definitions = shared + [
        (aliases[helper.symbol], expression)
        for helper, expression in zip(branch.helpers, helper_values, strict=True)
    ]
    if definitions:
        lines.append("  Definitions:")
        for symbol, expression in _ordered_definitions(definitions):
            lines.extend(_definition_lines(symbol, expression))

    if root_constraints:
        lines.append("  Polynomial root constraints:")
        for number, expression in enumerate(root_constraints, start=1):
            lines.append(f"    (R{number}) 0 =")
            lines.extend(_expression_lines(expression, "      "))

    if balances:
        lines.append("  Balance equations:")
        for number, expression in enumerate(balances, start=1):
            lines.append(f"    ({number}) 0 =")
            lines.extend(_expression_lines(expression, "      "))

    remaining_nonzero = tuple(
        expression
        for expression in branch.nonzero
        if expression != heading_nonzero
    )
    conditions = [
        _nonzero_text(expression.xreplace(aliases), result)
        for expression in remaining_nonzero
    ] + [_text(condition.xreplace(aliases)) for condition in branch.conditions]
    if conditions:
        lines.append("  Conditions:")
        lines.extend(f"    {condition}" for condition in conditions)
    return lines


def _outcome(result: EquilibriumRelation) -> CheckOutcome:
    details = [
        f"The complete physical steady-state relationship has "
        f"{_plural(len(result.branches), 'branch', 'branches')}.",
    ]
    if not result.branches:
        details.append("No physical steady state satisfies the source equations.")
    for index, branch in enumerate(result.branches, start=1):
        details.extend(_branch_details(branch, index, result))
    details.append("All branches are restricted to the configured physical bounds.")
    if result.diagnostic:
        details.append(result.diagnostic)

    return CheckOutcome(
        status=CheckStatus.PASS,
        details=tuple(details),
    )


def run(case: Case, _context: CheckContext) -> CheckOutcome:
    return _outcome(check_equilibria(case))


CHECK = CheckDefinition(
    id="equilibria",
    name="Equilibrium relationships",
    group="Network analysis",
    scope=CheckScope.CASE,
    run=run,
)
