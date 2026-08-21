"""Structural evaluation-cost profiling for DAETools-style kinetics trees."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache

import sympy as sp

from ..model import Reaction
from ..network import source_equivalent_fluxes


OPERATION_ORDER = (
    "add",
    "multiply",
    "reciprocal",
    "integer_power",
    "general_power",
    "sqrt",
    "exp",
    "log",
    "abs",
    "min",
    "max",
    "piecewise",
    "other_function",
)
_TRANSCENDENTAL = frozenset(("exp", "log", "sqrt", "general_power"))
_SWITCH = frozenset(("abs", "min", "max", "piecewise"))


class _FrozenMapping(Mapping[str, object]):
    """Small ordered, immutable and hashable mapping for profile metadata."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[tuple[str, object]]) -> None:
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("Frozen mappings cannot be modified.")

    def __getitem__(self, key: str) -> object:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items))


@dataclass(frozen=True)
class ExpressionStats:
    operations: tuple[tuple[str, int], ...]
    tree_nodes: int
    unique_nodes: int
    depth: int
    concentration_dependencies: tuple[sp.Symbol, ...]
    operating_dependencies: tuple[sp.Symbol, ...]
    ad_work: int
    unsupported_functions: tuple[str, ...]
    unsupported_subexpressions: tuple[sp.Expr, ...] = ()
    piecewise_branches: int = 0

    @property
    def total_operations(self) -> int:
        return sum(value for _, value in self.operations)

    @property
    def transcendental_operations(self) -> int:
        return sum(value for name, value in self.operations if name in _TRANSCENDENTAL)

    @property
    def switch_operations(self) -> int:
        return sum(value for name, value in self.operations if name in _SWITCH)

    @property
    def dae_dependencies(self) -> tuple[sp.Symbol, ...]:
        return _ordered_symbols(
            (*self.concentration_dependencies, *self.operating_dependencies)
        )

    @property
    def structural_jacobian_entries(self) -> int:
        return len(self.dae_dependencies)


@dataclass(frozen=True)
class CSEStats:
    operations: tuple[tuple[str, int], ...]
    temporary_count: int
    peak_live_temporaries: int

    @property
    def total_operations(self) -> int:
        return sum(value for _, value in self.operations)

    @property
    def transcendental_operations(self) -> int:
        return sum(value for name, value in self.operations if name in _TRANSCENDENTAL)

    @property
    def switch_operations(self) -> int:
        return sum(value for name, value in self.operations if name in _SWITCH)


@dataclass(frozen=True)
class SharedTerm:
    expression: sp.Expr
    operations: int
    occurrences: int
    outputs: tuple[str, ...]
    estimated_saved_operations: int


@dataclass(frozen=True)
class EvaluationProfile:
    declared_outputs: tuple[tuple[str, ExpressionStats], ...]
    flux_outputs: tuple[tuple[str, ExpressionStats], ...]
    declared_cse: CSEStats
    flux_cse: CSEStats
    declared_local_cse: tuple[tuple[str, CSEStats], ...]
    flux_local_cse: tuple[tuple[str, CSEStats], ...]
    flux_groups: tuple[Mapping[str, object], ...]
    shared_terms: tuple[SharedTerm, ...]
    declared_source_nnz: int
    flux_source_nnz: int


def _ordered_symbols(symbols: Iterable[sp.Symbol]) -> tuple[sp.Symbol, ...]:
    return tuple(sorted(set(symbols), key=sp.default_sort_key))


def _histogram(counter: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((name, int(counter.get(name, 0))) for name in OPERATION_ORDER)


def _operation(node: sp.Basic) -> tuple[str | None, int]:
    if isinstance(node, sp.Add):
        return "add", max(0, len(node.args) - 1)
    if isinstance(node, sp.Mul):
        return "multiply", max(0, len(node.args) - 1)
    if isinstance(node, sp.Pow):
        if node.exp == -1:
            return "reciprocal", 1
        if node.exp == sp.Rational(1, 2):
            return "sqrt", 1
        if node.exp.is_Integer:
            return "integer_power", 1
        return "general_power", 1
    if node.func is sp.exp:
        return "exp", 1
    if node.func is sp.log:
        return "log", 1
    if node.func is sp.Abs:
        return "abs", 1
    if node.func is sp.Min:
        return "min", max(0, len(node.args) - 1)
    if node.func is sp.Max:
        return "max", max(0, len(node.args) - 1)
    if isinstance(node, sp.Piecewise):
        return "piecewise", 1
    if node.is_Function:
        return "other_function", 1
    return None, 0


def _unsupported_name(node: sp.Basic) -> str | None:
    if isinstance(node, sp.Pow) and node.exp.is_number is not True:
        return "Pow"
    if isinstance(node, sp.Piecewise):
        return "Piecewise"
    if node.is_Function and node.func not in {
        sp.exp,
        sp.log,
        sp.Abs,
        sp.Min,
        sp.Max,
    }:
        return node.func.__name__
    return None


def _profile_expression(
    expression: sp.Expr,
    concentration_symbols: frozenset[sp.Symbol],
    operating_symbols: frozenset[sp.Symbol],
) -> ExpressionStats:
    dae_symbols = concentration_symbols | operating_symbols

    @cache
    def dependencies(node: sp.Basic) -> frozenset[sp.Symbol]:
        if isinstance(node, sp.Symbol):
            return frozenset((node,)) if node in dae_symbols else frozenset()
        return frozenset().union(*(dependencies(arg) for arg in node.args))

    @cache
    def depth(node: sp.Basic) -> int:
        return 1 + max((depth(arg) for arg in node.args), default=0)

    operations: Counter[str] = Counter()
    ad_work = 0
    unsupported: dict[sp.Expr, str] = {}
    branches = 0
    nodes = tuple(sp.preorder_traversal(expression))
    for node in nodes:
        name, multiplicity = _operation(node)
        if name is not None:
            operations[name] += multiplicity
            ad_work += multiplicity * len(dependencies(node))
        unsupported_name = _unsupported_name(node)
        if unsupported_name is not None:
            unsupported.setdefault(node, unsupported_name)
        if isinstance(node, sp.Piecewise):
            branches += len(node.args)
    free = expression.free_symbols
    unsupported_nodes = tuple(unsupported)
    return ExpressionStats(
        _histogram(operations),
        len(nodes),
        len(set(nodes)),
        depth(expression),
        _ordered_symbols(free & concentration_symbols),
        _ordered_symbols(free & operating_symbols),
        ad_work,
        tuple(dict.fromkeys(unsupported.values())),
        unsupported_nodes,
        branches,
    )


def _sum_operations(stats: Iterable[ExpressionStats | CSEStats]) -> Counter[str]:
    total: Counter[str] = Counter()
    for item in stats:
        total.update(dict(item.operations))
    return total


def _peak_liveness(
    replacements: Sequence[tuple[sp.Symbol, sp.Expr]],
    outputs: Sequence[sp.Expr],
) -> int:
    temporaries = frozenset(symbol for symbol, _ in replacements)
    last_use: dict[sp.Symbol, int] = {}
    expressions = (*tuple(value for _, value in replacements), *outputs)
    for position, expression in enumerate(expressions):
        for symbol in expression.free_symbols & temporaries:
            last_use[symbol] = position

    active: set[sp.Symbol] = set()
    peak = 0
    for position, (symbol, _value) in enumerate(replacements):
        active.add(symbol)
        peak = max(peak, len(active))
        active.difference_update(
            item for item in tuple(active) if last_use.get(item) == position
        )
    return peak


def _cse_stats(expressions: Sequence[sp.Expr]) -> CSEStats:
    replacements, outputs = sp.cse(
        tuple(expressions),
        symbols=sp.numbered_symbols("t"),
        order="canonical",
    )
    profiled = tuple(
        _profile_expression(expression, frozenset(), frozenset())
        for expression in (*tuple(value for _, value in replacements), *outputs)
    )
    return CSEStats(
        _histogram(_sum_operations(profiled)),
        len(replacements),
        _peak_liveness(replacements, outputs),
    )


def _shared_terms(
    outputs: Sequence[tuple[str, sp.Expr]],
) -> tuple[SharedTerm, ...]:
    occurrences: Counter[sp.Expr] = Counter()
    users: defaultdict[sp.Expr, list[str]] = defaultdict(list)
    for output_id, expression in outputs:
        local = Counter(sp.preorder_traversal(expression))
        for node, count in local.items():
            if node.is_Atom:
                continue
            occurrences[node] += count
            users[node].append(output_id)

    terms = []
    for expression, count in occurrences.items():
        if count < 2:
            continue
        operations = _profile_expression(
            expression, frozenset(), frozenset()
        ).total_operations
        if not operations:
            continue
        terms.append(
            SharedTerm(
                expression,
                operations,
                count,
                tuple(users[expression]),
                (count - 1) * operations,
            )
        )
    return tuple(
        sorted(
            terms,
            key=lambda item: (
                -item.estimated_saved_operations,
                -item.operations,
                -item.occurrences,
                sp.default_sort_key(item.expression),
            ),
        )
    )


def profile_evaluation(
    reactions: Sequence[Reaction],
    stoichiometry: sp.MatrixBase,
    concentration_symbols: Iterable[sp.Symbol],
    operating_symbols: Iterable[sp.Symbol],
) -> EvaluationProfile:
    """Profile mathematical work per cell and residual evaluation."""

    reactions = tuple(reactions)
    if stoichiometry.cols != len(reactions):
        raise ValueError("Stoichiometry columns must match the reactions.")
    concentration_set = frozenset(concentration_symbols)
    operating_set = frozenset(operating_symbols)
    if concentration_set & operating_set:
        raise ValueError("Concentration and operating symbols must be distinct.")

    declared = tuple((reaction.id, reaction.rate) for reaction in reactions)
    network = source_equivalent_fluxes(reactions, stoichiometry)
    fluxes = tuple((flux.id, flux.expression) for flux in network.fluxes)
    flux_groups = tuple(
        _FrozenMapping(
            (
                ("id", flux.id),
                ("stoichiometry", flux.stoichiometry),
                ("members", flux.members),
                ("expression", flux.expression),
            )
        )
        for flux in network.fluxes
    )
    declared_stats = tuple(
        (output_id, _profile_expression(expression, concentration_set, operating_set))
        for output_id, expression in declared
    )
    flux_stats = tuple(
        (output_id, _profile_expression(expression, concentration_set, operating_set))
        for output_id, expression in fluxes
    )
    return EvaluationProfile(
        declared_stats,
        flux_stats,
        _cse_stats(tuple(expression for _, expression in declared)),
        _cse_stats(tuple(expression for _, expression in fluxes)),
        tuple((output_id, _cse_stats((expression,))) for output_id, expression in declared),
        tuple((output_id, _cse_stats((expression,))) for output_id, expression in fluxes),
        flux_groups,
        _shared_terms(declared),
        sum(value != 0 for value in stoichiometry),
        sum(value != 0 for value in network.stoichiometry),
    )


__all__ = (
    "CSEStats",
    "EvaluationProfile",
    "ExpressionStats",
    "OPERATION_ORDER",
    "SharedTerm",
    "profile_evaluation",
)
