"""Structural evaluation-cost profiling for DAETools-style kinetics trees."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType

import sympy as sp

from ..model import Reaction
from ..network import SourceFlux, source_equivalent_fluxes

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


@dataclass(frozen=True)
class OperationStats:
    operations: Mapping[str, int]
    total_operations: int = field(init=False)
    transcendental_operations: int = field(init=False)
    switch_operations: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))
        object.__setattr__(self, "total_operations", sum(self.operations.values()))
        object.__setattr__(
            self,
            "transcendental_operations",
            sum(value for name, value in self.operations.items() if name in _TRANSCENDENTAL),
        )
        object.__setattr__(
            self,
            "switch_operations",
            sum(value for name, value in self.operations.items() if name in _SWITCH),
        )


@dataclass(frozen=True)
class ExpressionStats(OperationStats):
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
    def dae_dependencies(self) -> tuple[sp.Symbol, ...]:
        return _ordered_symbols((*self.concentration_dependencies, *self.operating_dependencies))

    @property
    def structural_jacobian_entries(self) -> int:
        return len(self.dae_dependencies)


@dataclass(frozen=True)
class CSEStats(OperationStats):
    temporary_count: int
    peak_live_temporaries: int


@dataclass(frozen=True)
class SharedTerm:
    expression: sp.Expr
    operations: int
    occurrences: int
    outputs: tuple[str, ...]
    estimated_saved_operations: int


@dataclass(frozen=True)
class EvaluationView:
    outputs: Mapping[str, ExpressionStats]
    raw: OperationStats
    cse: CSEStats
    local_cse: Mapping[str, CSEStats]
    source_nnz: int
    rate_input_entries: int
    residual_entries: int
    groups: tuple[SourceFlux, ...] = ()


@dataclass(frozen=True)
class EvaluationProfile:
    declared: EvaluationView
    source_equivalent: EvaluationView
    shared_terms: tuple[SharedTerm, ...]


def _ordered_symbols(symbols) -> tuple[sp.Symbol, ...]:
    return tuple(sorted(set(symbols), key=sp.default_sort_key))


def _histogram(counter) -> dict[str, int]:
    return {name: counter.get(name, 0) for name in OPERATION_ORDER}


def _operation(node) -> tuple[str | None, int]:
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


def _unsupported_name(node) -> str | None:
    if isinstance(node, sp.Pow) and node.exp.is_number is not True:
        return "Pow"
    if isinstance(node, sp.Piecewise):
        return "Piecewise"
    if node.is_Function and node.func not in {sp.exp, sp.log, sp.Abs, sp.Min, sp.Max}:
        return node.func.__name__
    return None


def _profile_expression(expression, concentration_symbols, operating_symbols) -> ExpressionStats:
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


def _operation_counts(expressions):
    counts = Counter()
    for expression in expressions:
        for node in sp.preorder_traversal(expression):
            name, multiplicity = _operation(node)
            if name is not None:
                counts[name] += multiplicity
    return _histogram(counts)


def _peak_liveness(replacements, outputs) -> int:
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
        active.difference_update(item for item in tuple(active) if last_use.get(item) == position)
    return peak


def _cse_stats(expressions) -> CSEStats:
    replacements, outputs = sp.cse(
        tuple(expressions), symbols=sp.numbered_symbols("t"), order="canonical"
    )
    expressions = [value for _, value in replacements] + list(outputs)
    return CSEStats(
        _operation_counts(expressions), len(replacements), _peak_liveness(replacements, outputs)
    )


def _shared_terms(outputs) -> tuple[SharedTerm, ...]:
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
        operations = sum(_operation_counts((expression,)).values())
        if not operations:
            continue
        terms.append(
            SharedTerm(
                expression, operations, count, tuple(users[expression]), (count - 1) * operations
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

    def view(outputs, matrix, groups=()):
        stats = {
            name: _profile_expression(expression, concentration_set, operating_set)
            for name, expression in outputs
        }
        counts = Counter()
        for item in stats.values():
            counts.update(item.operations)
        source_nnz = sum(value != 0 for value in matrix)
        inputs = sum(item.structural_jacobian_entries for item in stats.values())
        return EvaluationView(
            stats,
            OperationStats(_histogram(counts)),
            _cse_stats(tuple(expression for _, expression in outputs)),
            {name: _cse_stats((expression,)) for name, expression in outputs},
            source_nnz,
            inputs,
            len(outputs) + inputs + source_nnz,
            groups,
        )

    return EvaluationProfile(
        view(declared, stoichiometry),
        view(fluxes, network.stoichiometry, network.fluxes),
        _shared_terms(declared),
    )
